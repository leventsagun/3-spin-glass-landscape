require 'math'
-- require 'cunn'
require 'nn'
-- require 'cutorch'
require 'torch'
require 'optim'
--------------------------------------------------
require '3spin'

torch.setdefaulttensortype("torch.DoubleTensor")
-- cutorch.setDevice(4) --start counting from 1
torch.manualSeed(1)

M = 1 --#samples
N = 200 --dim
dist = 1 --(last one gives nan)
num=1 --experiment#
while paths.dirp('trial_'..dist..'_'..N..'_'..num) do
  num=num+1
end
path = '/Users/leventsagun/Dropbox/calismalar/code/spin-glass/code/results_'..dist..'_'..N..'_'..num..'/'

learningRate = 0.2
learningRateDecay = 0.00
ValuePrecision = 0
precision = {0.01}--, 3, 2, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.175, 0.15, 0.125, 0.1, 0.09} --0.08 , 0.06, 0.04, 0.02, 0.01} --in decreasing order

timekeeper = optim.Logger(paths.concat(path, 'times.log'))
energies = optim.Logger(paths.concat(path, 'energies.log'))
setup=optim.Logger(paths.concat(path, 'setup.log'))
setup:add{['setup']='gradient descent on a random initial pt (annealed)'}
setup:add{['setup']='learningRate: '..learningRate}
setup:add{['setup']='dimension: '..N}
setup:add{['setup']='init dist: '..dist}
prec = ""
for j  = 1, #precision do
   prec = prec .. " " .. precision[j]
end
setup:add{['setup']='precision: '..prec} --adds it side by side
setup:add{['setup']='#precision: '..#precision}

for i=1,M do
    Surface = FixEnv(N, dist) --annealed 1 Gaussian
    SpinItGD(Surface)
    for l=1,#precision do
      timekeeper:add{['HaltingTime'] = Times[l]}
      energies:add{['Energies'] = Energies[l]}
    end
    collectgarbage()
end
