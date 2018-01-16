require 'torch'
require 'math'
require 'cunn'
torch.setdefaulttensortype("torch.FloatTensor")

function Spin(N, M, P, l)

cutorch.setDevice(l)

--torch.manualSeed(1) -- so the surface can be regenarated 

Values = torch.CudaTensor(M)

w = torch.CudaTensor(N)
w2 = torch.CudaTensor(N)
X_sample = torch.CudaTensor(P, N, N, N)
H = torch.CudaTensor(N, N, N)
X = torch.CudaTensor(N, N, N)
V = torch.CudaTensor(N, N, N)


gw = torch.CudaTensor(N)
ger_tmp = torch.CudaTensor(N, N)
ger_tmp2 = torch.CudaTensor(N^2, N)

learningRate = 0.3 --was 0.3 for GD
nIterations = 500
learningRateDecay = 0.0015 -- was 0.0015 for GD

paths = torch.CudaTensor(M, nIterations/10 + 1 + 100)

X_sample:normal():div(math.sqrt(P))
X=X_sample:sum(1)
X:resize(N, N, N)

Grad = torch.CudaTensor(M)

for trial = 1, M do

j=1 -- counter for the paths
--torch.manualSeed(123) 
w:normal()
w:mul(math.sqrt(N)/(w:norm()))

    w2:copy(w):resize(N, 1)
    w3 = w:resize(1, N)
    ger_tmp:resize(N, N)
    ger_tmp:addmm(0, 1, w2, w3)
    w:resize(N)
    ger_tmp2:addmm(0, 1, ger_tmp:resize(N^2, 1), w:resize(1, N))
    H:copy(X_sample[1])
    H:resize(N, N, N)
    H:cmul(ger_tmp2)

    V:copy(X)
    V:cmul(ger_tmp2)    
    --print("value: " .. V:sum()/N)
    paths[trial][j]=V:sum()/N
    j=j+1

    for iIteration = 1,nIterations*P do

        gamma = learningRate / (1 + iIteration * learningRateDecay) 
        gw:zero()
        gw:add(H:sum(1):sum(3)):add(H:sum(2):sum(1)):add(H:sum(3):sum(2))
        gw:cdiv(w):div(N)
        gw:add(-gw:dot(w)/N, w)
        w:add(-gamma, gw)
        w:div(w:norm()/math.sqrt(N))

        
        w2:copy(w):resize(N, 1)
        w3 = w:resize(1, N)
        ger_tmp:resize(N, N)
        ger_tmp:addmm(0, 1, w2, w3)
        ger_tmp2:addmm(0, 1, ger_tmp:resize(N^2, 1), w:resize(1, N))
        H:copy(X_sample[1+iIteration%P])
        H:resize(N, N, N)
        H:cmul(ger_tmp2)
        w:resize(N)

        if iIteration%(10*P) == 0 then 
            V:copy(X)
            V:cmul(ger_tmp2)    
            paths[trial][j]=V:sum()/N
            j=j+1

            V:copy(X)
            V:cmul(ger_tmp2)           
            gw:zero()
            gw:add(V:sum(1):sum(3)):add(V:sum(2):sum(1)):add(V:sum(3):sum(2))
            gw:cdiv(w):div(N)
            gw:add(-gw:dot(w)/N, w)
            print("value: " .. V:sum()/N)
            print("size of grad: " .. gw:norm())
        end 

        collectgarbage()
    end

    print("end of SGD: ")
    print(" ")
    
    for k =1, 1000 do
        gamma = 0.2 / (1 + k * learningRateDecay) 
        

        gw:zero()
        gw:add(V:sum(1):sum(3)):add(V:sum(2):sum(1)):add(V:sum(3):sum(2))
        gw:cdiv(w):div(N)
        gw:add(-gw:dot(w)/N, w)
        w:add(-gamma, gw)
        w:div(w:norm()/math.sqrt(N))

        
        w2:copy(w):resize(N, 1)
        w3 = w:resize(1, N)
        ger_tmp:resize(N, N)
        ger_tmp:addmm(0, 1, w2, w3)
        ger_tmp2:addmm(0, 1, ger_tmp:resize(N^2, 1), w:resize(1, N))
        V:copy(X)
        V:cmul(ger_tmp2)           
        w:resize(N)

        if k%10 == 0 then
            paths[trial][j]=V:sum()/N
            j=j+1
            print("value: " .. V:sum()/N)
            print("size of grad: " .. gw:norm())
        end

        collectgarbage()


    end
    
    Grad[trial]=gw:norm()
    Values[trial] = V:sum()/N
    

end

file = torch.DiskFile('/home/sagun/SpinGlass/SGD_values_' .. N .. '_' .. M .. '_' .. P .. '.asc', 'w')
file:writeObject(Values)
file:close()

file = torch.DiskFile('/home/sagun/SpinGlass/SGD_paths_' .. N .. '_' .. M .. '_' .. P .. '.asc', 'w')
file:writeObject(paths)
file:close()

file = torch.DiskFile('/home/sagun/SpinGlass/SGD_grad_' .. N .. '_' .. M .. '_' .. P .. '.asc', 'w')
file:writeObject(Grad)
file:close()

end
