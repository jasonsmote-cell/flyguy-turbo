-- flyguy (github.com/adventure-gpt/flyguy, MIT). Forward pass only.
-- Sparse connectome walk: only neurons that fire are walked, so silence is free.
local Fly = {}
local W = ...
local buf, O = nil, nil
local sz, HOPS, OBS = nil, 0, 0
local rd = buffer.readf32
local vzero = vector.zero

local function relu(x) return x > 0 and x or 0 end
local function f(off, i) return rd(buf, off + i * 4) end
local function i(off, j) return rd(buf, off + j * 4) end

local drive, hA, hB, zA

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
	local cw, cc, cb = o.in_w, o.in_c, o.in_b
	local icr = o.in_crow
	for r = 0, n0 - 1 do
		local acc = 0
		local a, b = f(icr, r), f(icr, r + 1) - 1
		for j = a, b do acc += f(cw, j) * obs[i(cc, j)] end
		drive[r] = acc + f(cb, r)
	end
	self.step()
	local h = hA
	for r = 0, n0 - 1 do h[r] = relu(f(o.g0, r) * drive[r] - f(o.t0, r)) end
	for k = 1, HOPS do
		local prev, rows = sz[k - 1], sz[k]
		local z, h2 = zA, hB
		for r = 0, rows - 1 do z[r] = 0 end
		local tcr, tco, tw = o["tcrow" .. k], o["tcol" .. k], o["tw" .. k]
		for s = 0, prev - 1 do
			local hs = h[s]
			if hs > 0 then
				local a, b = f(tcr, s), f(tcr, s + 1) - 1
				for j = a, b do
					local t = i(tco, j)
					z[t] += f(tw, j) * hs
				end
			end
		end
		self.step()
		local sn, sOff, dOff = o["s" .. k .. "_n"], o["s" .. k], o["d" .. k]
		for j = 0, sn - 1 do z[i(sOff, j)] += drive[i(dOff, j)] end
		local g, t = o["g" .. k], o["t" .. k]
		for r = 0, rows - 1 do h2[r] = relu(f(g, r) * z[r] - f(t, r)) end
		h, hA, hB = h2, h2, h
	end
	local dn, ow, ob = o.dn, o.ow, o.ob
	local nd = sz[HOPS]
	local best, bestV = 1, -math.huge
	for a = 1, 90 do
		local acc = f(ob, a - 1)
		local base = (a - 1) * nd
		for j = 0, nd - 1 do acc += f(ow, base + j) * h[j] end
		if acc > bestV then best, bestV = a, acc end
	end
	return best
end
