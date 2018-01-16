require 'torch'
require 'math'
require 'cunn'
torch.setdefaulttensortype("torch.FloatTensor")

function USpin(N, M, l)

cutorch.setDevice(l)

torch.manualSeed(1) -- so the surface can be regenarated 

States = torch.CudaTensor(M, 3*N)
Values = torch.CudaTensor(M)

w1 = torch.CudaTensor(N) -- points on the sphere
w2 = torch.CudaTensor(N)
w3 = torch.CudaTensor(N)
w = torch.CudaTensor(3*N) -- w = (w1, w2, w3)
gw = torch.CudaTensor(3*N) -- gradient will be bigger now
gw1 = torch.CudaTensor(N) -- gradient will be bigger now
gw2 = torch.CudaTensor(N) -- gradient will be bigger now
gw3 = torch.CudaTensor(N) -- gradient will be bigger now

w1 = w:narrow(1, 1, N)
w2 = w:narrow(1, N+1, N)
w3 = w:narrow(1, 2*N+1, N)

gw1 = gw:narrow(1, 1, N)
gw2 = gw:narrow(1, N+1, N)
gw3 = gw:narrow(1, 2*N+1, N)

X = torch.CudaTensor(N, N, N) -- X for coupling tensor
H = torch.CudaTensor(N, N, N) -- H for Hamiltonian value

ger_tmp = torch.CudaTensor(N, N)
ger_tmp2 = torch.CudaTensor(N^2, N)

learningRate = 0.3 --was 0.3
nIterations = 1000
learningRateDecay = 0.0015
X:normal()

for trial = 1, M do

    --three random points on the sphere OR one point on the product of three spheres
    w1:normal():mul(math.sqrt(N)/(w1:norm()))
    w2:normal():mul(math.sqrt(N)/(w2:norm()))
    w3:normal():mul(math.sqrt(N)/(w3:norm()))

    w1:resize(N, 1)
    w2:resize(1, N)
    ger_tmp:resize(N, N)
    ger_tmp:addmm(0, 1, w1, w2)
    ger_tmp2:addmm(0, 1, ger_tmp:resize(N^2, 1), w3:resize(1, N))
    H:copy(X)
    H:cmul(ger_tmp2)
    
    w1:resize(N)
    w2:resize(N)
    w3:resize(N)

    for iIteration = 1,nIterations do

        gamma = learningRate / (1 + iIteration * learningRateDecay) 
        
        gw1:zero()
        gw2:zero()
        gw3:zero()
        
        gw2:add(H:sum(1):sum(3))
        gw3:add(H:sum(2):sum(1))
        gw1:add(H:sum(3):sum(2))

        gw1:cdiv(w1):div(N)
        gw2:cdiv(w2):div(N)
        gw3:cdiv(w3):div(N)
        
        gw1:add(-gw1:dot(w1)/N, w1)
        gw2:add(-gw2:dot(w2)/N, w2)
        gw3:add(-gw3:dot(w3)/N, w3)

        w:add(-gamma, gw)
        
        w1:div(w1:norm()/math.sqrt(N))
        w2:div(w2:norm()/math.sqrt(N))
        w3:div(w3:norm()/math.sqrt(N))

        
        w1:resize(N, 1)
        w2:resize(1, N)
        ger_tmp:resize(N, N)
        ger_tmp:addmm(0, 1, w1, w2)
        ger_tmp2:addmm(0, 1, ger_tmp:resize(N^2, 1), w3:resize(1, N))
        H:copy(X)
        H:cmul(ger_tmp2)
      
        w1:resize(N)
        w2:resize(N)
        w3:resize(N)

        collectgarbage()
    end

    States[trial] = w
    Values[trial] = H:sum()/N
    print("trial: " .. trial)
        
    end

    --file name begins with U for uncoupled 

    file = torch.DiskFile('/home/sagun/SpinGlass/U_states_' .. N .. '_' .. M .. '.asc', 'w')
    file:writeObject(States)
    file:close()

    file = torch.DiskFile('/home/sagun/SpinGlass/U_values_' .. N .. '_' .. M .. '.asc', 'w')
    file:writeObject(Values)
    file:close()

end