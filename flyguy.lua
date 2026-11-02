-- flyguy (github.com/adventure-gpt/flyguy, MIT). Forward pass + observation only.
-- Sparse connectome walk: only neurons that fire are walked, so silence costs nothing.
local Fly = {}

local POS_STD, VEL, BALL_VEL, ANG, BALL_ANG = { 4096, 5120, 2044 }, 2300, 6000, 5.5, 6.0
local DIST, BALL_R, GRAV, REST, DRAG = 5000, 92.75, 650, 0.6, 0.03
local FX, FY, FZ = 4096, 5120, 2044
local GOAL_Y, GOAL_Z = 5120, 321

Fly.PADS = { -- rlgym BOOST_LOCATIONS order, xyz
	{0,-4240,70},{-1792,-4184,70},{1792,-4184,70},{-3072,-4096,73},{3072,-4096,73},{-940,-3308,70},{940,-3308,70},{0,-2816,70},
	{-3584,-2484,70},{3584,-2484,70},{-1788,-2300,70},{1788,-2300,70},{-2048,-1036,70},{0,-1024,70},{2048,-1036,70},{-3584,0,73},
	{-1024,0,70},{1024,0,70},{3584,0,73},{-2048,1036,70},{0,1024,70},{2048,1036,70},{-1788,2300,70},{1788,2300,70},{-3584,2484,70},
	{3584,2484,70},{0,2816,70},{-940,3310,70},{940,3308,70},{-3072,4096,73},{3072,4096,73},{-1792,4184,70},{1792,4184,70},{0,4240,70},
}
Fly.PAD_CD = {}
for i = 1, 34 do Fly.PAD_CD[i] = (Fly.PADS[i][3] > 72) and 10 or 4 end
Fly.PAD_MIRROR = {}
for i = 1, 34 do Fly.PAD_MIRROR[i] = 35 - i end

local rd = buffer.readf32
local buf, O, sz, HOPS, OBS = nil, nil, nil, 0, 0
local drive, hA, hB, zA

local function relu(x) return x > 0 and x or 0 end
local function f(off, i) return rd(buf, off + i * 4) end
local function ix(off, i) return rd(buf, off + i * 4) end

function Fly.load(w)
	buf = buffer.fromstring(crypt.base64decode(w.data))
	O, sz, HOPS, OBS = w.off, w.sizes, w.hops, w.obs_dim
	hA, hB = table.create(sz[HOPS]), table.create(sz[HOPS])
	zA = table.create(sz[HOPS])
	drive = table.create(sz[0])
	return setmetatable({}, { __index = Fly })
end

function Fly.step() coroutine.yield() end

function Fly:forward(obs)
	local n0, o = sz[0], O
	local cw, cc, cb, icr = o.in_w, o.in_c, o.in_b, o.in_crow
	for r = 0, n0 - 1 do
		local acc, a, b = 0, f(icr, r), f(icr, r + 1) - 1
		for j = a, b do acc += f(cw, j) * obs[ix(cc, j)] end
		drive[r] = acc + f(cb, r)
	end
	self.step()
	local h = hA
	for r = 0, n0 - 1 do h[r] = relu(f(o.g0, r) * drive[r] - f(o.t0, r)) end
	for k = 1, HOPS do
		local prev, rows, z, h2 = sz[k - 1], sz[k], zA, hB
		for r = 0, rows - 1 do z[r] = 0 end
		local tcr, tco, tw = o["tcrow" .. k], o["tcol" .. k], o["tw" .. k]
		for s = 0, prev - 1 do
			local hs = h[s]
			if hs > 0 then
				local a, b = f(tcr, s), f(tcr, s + 1) - 1
				for j = a, b do
					local t = ix(tco, j)
					z[t] += f(tw, j) * hs
				end
			end
		end
		self.step()
		local sn, sOff, dOff = o["s" .. k .. "_n"], o["s" .. k], o["d" .. k]
		for j = 0, sn - 1 do z[ix(sOff, j)] += drive[ix(dOff, j)] end
		local g, t = o["g" .. k], o["t" .. k]
		for r = 0, rows - 1 do h2[r] = relu(f(g, r) * z[r] - f(t, r)) end
		hA, hB, h = hB, hA, hB
	end
	local dn, ow, ob, nd = o.dn, o.ow, o.ob, sz[HOPS]
	local best, bestV = 1, -math.huge
	for a = 1, 90 do
		local acc, base = f(ob, a - 1), (a - 1) * nd
		for j = 0, nd - 1 do acc += f(ow, base + j) * h[j] end
		if acc > bestV then best, bestV = a, acc end
	end
	return best
end

---------------------------------------------------------------- observation (flyguy/obs.py, 517 numbers)
local function fold(x, lim)
	local m = (x + lim) % (4 * lim)
	if m < 2 * lim then return m - lim end
	return 3 * lim - m
end

local function ballPath(px, py, pz, vx, vy, vz, t, out, at)
	local fq = (1 - math.exp(-DRAG * t)) / DRAG
	local z0 = math.max(pz, BALL_R)
	local tb = (vz + math.sqrt(vz * vz + 2 * GRAV * (z0 - BALL_R))) / GRAV
	local zf = z0 + vz * t - 0.5 * GRAV * t * t
	local rb = -REST * (vz - GRAV * tb)
	local s = math.max(t - tb, 0)
	local z = (t < tb) and zf or (BALL_R + rb * s - 0.5 * GRAV * s * s)
	out[at] = fold(px + vx * fq, FX - BALL_R) / FX
	out[at + 1] = fold(py + vy * fq, FY - BALL_R) / FY
	out[at + 2] = math.clamp(z, BALL_R, FZ - BALL_R) / FZ
end

-- arena: cars with GetState, ball with GetState, boostPads. mem: {action, jumped, padMap, padArena, lat}
function Fly.buildObs(arena, me, mem)
	local invert = me.team == 1
	local lat = mem.lat or 0
	if mem.padArena ~= arena then mem.padArena, mem.padMap = arena, nil end
	if not mem.padMap then
		mem.padMap = {}
		for i, L in Fly.PADS do
			local best, bd
			for j, pad in arena.boostPads do
				local d = (pad.pos.x - L[1]) ^ 2 + (pad.pos.y - L[2]) ^ 2
				if not bd or d < bd then best, bd = j, d end
			end
			mem.padMap[i] = best
		end
	end

	local o = table.create(OBS)
	local s = invert and -1 or 1
	local st = me:GetState()
	local p, v, w, fw, up = st.pos, st.vel, st.angVel, st.rot.forward, st.rot.up
	local b = arena.ball:GetState()
	local rx, ry = up.y * fw.z - up.z * fw.y, up.z * fw.x - up.x * fw.z
	local rz = up.x * fw.y - up.y * fw.x
	local px, py, pz = p.x * s, p.y * s, p.z
	local vx, vy, vz = v.x * s, v.y * s, v.z
	local wx, wy, wz = w.x * s, w.y * s, w.z
	local fx, fy, fz = fw.x * s, fw.y * s, fw.z
	local ux, uy, uz = up.x * s, up.y * s, up.z
	local r_x, r_y, r_z = rx * s, ry * s, rz
	local bx, by, bz = (b.pos.x + b.vel.x * lat) * s, (b.pos.y + b.vel.y * lat) * s, b.pos.z + b.vel.z * lat
	local bvx, bvy, bvz = b.vel.x * s, b.vel.y * s, b.vel.z
	local relx, rely, relz = bx - px, by - py, bz - pz
	local rvx, rvy, rvz = bvx - vx, bvy - vy, bvz - vz
	local function loc(a, c, d, o1, o2, o3)
		return a * fx + c * fy + d * fz, a * r_x + c * r_y + d * r_z, a * ux + c * uy + d * uz
	end
	local i = 1
	local function w3(a, b2, c) o[i] = a; o[i + 1] = b2; o[i + 2] = c; i = i + 3 end

	w3(px / POS_STD[1], py / POS_STD[2], pz / POS_STD[3])
	w3(vx / VEL, vy / VEL, vz / VEL)
	w3(wx / ANG, wy / ANG, wz / ANG)
	w3(fx, fy, fz); w3(ux, uy, uz)
	local a1, b1, c1 = loc(vx, vy, vz); w3(a1 / VEL, b1 / VEL, c1 / VEL)
	local a2, b2, c2 = loc(wx, wy, wz); w3(a2 / ANG, b2 / ANG, c2 / ANG)
	o[i] = st.boost / 100; i = i + 1
	local onG = st.isOnGround and 1 or 0
	if st.isOnGround then mem.jumped[me.id] = false elseif not mem.jumped[me.id] and st.vel.z > 40 then mem.jumped[me.id] = true end
	local jumped = mem.jumped[me.id] and 1 or 0
	local can = (not st.isOnGround) and not st.hasFlipped and not st.hasDoubleJumped
	o[i] = onG; o[i+1] = jumped; o[i+2] = st.hasDoubleJumped and 1 or 0; o[i+3] = st.hasFlipped and 1 or 0
	o[i+4] = (not st.isOnGround and not st.isFlipping and st.vel.z > 50) and 1 or 0
	o[i+5] = st.isFlipping and 1 or 0; o[i+6] = can and 1 or 0
	o[i+7] = (math.sqrt(v.x^2 + v.y^2) >= 2200) and 1 or 0; o[i+8] = st.isDemoed and 1 or 0
	i = i + 9
	o[i] = can and 1.25 or 0; o[i+1] = st.isDemoed and 3 or 0; i = i + 2
	w3(bx / POS_STD[1], by / POS_STD[2], bz / POS_STD[3])
	w3(bvx / BALL_VEL, bvy / BALL_VEL, bvz / BALL_VEL)
	w3(b.angVel.x * s / BALL_ANG, b.angVel.y * s / BALL_ANG, b.angVel.z / BALL_ANG)
	w3(relx / DIST, rely / DIST, relz / DIST)
	local r1, r2, r3 = loc(relx, rely, relz); w3(r1 / DIST, r2 / DIST, r3 / DIST)
	local q1, q2, q3 = loc(rvx, rvy, rvz); w3(rvx / VEL, rvy / VEL, rvz / VEL); w3(q1 / VEL, q2 / VEL, q3 / VEL)
	o[i] = math.sqrt(relx^2 + rely^2 + relz^2) / DIST; i = i + 1
	local g1, g2, g3 = loc(-px, GOAL_Y - py, GOAL_Z - pz); w3(g1 / DIST, g2 / DIST, g3 / DIST)
	local h1, h2, h3 = loc(-px, -GOAL_Y - py, GOAL_Z - pz); w3(h1 / DIST, h2 / DIST, h3 / DIST)
	for n = 1, 34 do
		local pad = arena.boostPads[mem.padMap[invert and Fly.PAD_MIRROR[n] or n]]
		o[i + n - 1] = (pad and pad.isActive) and 1 or 0
	end
	i = i + 34
	for n = 1, 34 do
		local pad = arena.boostPads[mem.padMap[invert and Fly.PAD_MIRROR[n] or n]]
		local cd = Fly.PAD_CD[n]
		local left = (pad and pad.isActive) and 0 or (pad and (pad.timer or pad.timeLeft or cd) or cd)
		o[i + n - 1] = left / 10
	end
	i = i + 34
	for n = 1, 8 do o[i + n - 1] = mem.action[n] end
	i = i + 8
	for n = 1, 34 do
		local L = Fly.PADS[invert and Fly.PAD_MIRROR[n] or n]
		local dx, dy, dz = L[1] - px, L[2] - py, L[3] - pz
		local p1, p2, p3 = loc(dx, dy, dz); w3(p1 / DIST, p2 / DIST, p3 / DIST)
	end
	for n = 1, 34 do
		local L = Fly.PADS[invert and Fly.PAD_MIRROR[n] or n]
		o[i + n - 1] = math.sqrt((L[1] - px)^2 + (L[2] - py)^2 + (L[3] - pz)^2) / DIST
	end
	i = i + 34
	for n = 1, 4 do ballPath(bx, by, bz, bvx, bvy, bvz, 0.5 * n, o, i); i = i + 3 end
	for n = 1, 4 do
		local t = 0.5 * n
		local tmp = table.create(3)
		ballPath(bx, by, bz, bvx, bvy, bvz, t, tmp, 1)
		local dx, dy, dz = tmp[1] * FX - px, tmp[2] * FY - py, tmp[3] * FZ - pz
		local p1, p2, p3 = loc(dx, dy, dz); w3(p1 / DIST, p2 / DIST, p3 / DIST)
	end
	return o
end

Fly.KICKOFF = {}
do
	local function add(n, a) for _ = 1, n * 4 do table.insert(Fly.KICKOFF, a) end end
	add(11, {1,0,0,0,0,0,1,0}) add(4, {1,-1,0,0,0,0,1,0}) add(2, {1,0,0,0,0,1,1,0}) add(1, {1,0,0,0,0,0,1,0})
	add(1, {1,0,-0.7,0.8,0,1,1,0}) add(13, {1,0,1,0,0,0,1,0}) add(10, {1,0,0.5,0,1,0,0,0})
end

do
	local A = {}
	for _, th in { -1, 0, 1 } do
		for _, st in { -1, 0, 1 } do
			for _, bo in { 0, 1 } do
				for _, hb in { 0, 1 } do
					if not (bo == 1 and th ~= 1) then table.insert(A, { (th ~= 0 and th) or bo, st, 0, st, 0, 0, bo, hb }) end
				end
			end
		end
	end
	for _, p in { -1, 0, 1 } do
		for _, y in { -1, 0, 1 } do
			for _, r in { -1, 0, 1 } do
				for _, j in { 0, 1 } do
					for _, bo in { 0, 1 } do
						if not (j == 1 and y ~= 0) and not (p == 0 and r == 0 and j == 0) then
							local hb = (j == 1 and (p ~= 0 or y ~= 0 or r ~= 0)) and 1 or 0
							table.insert(A, { bo, y, p, y, r, j, bo, hb })
						end
					end
				end
			end
		end
	end
	Fly.ACTIONS = A
end

return Fly
