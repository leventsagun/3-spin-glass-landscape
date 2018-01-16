--overlap matrix

function overlap(States, N, M)
	O = torch.CudaTensor(M, M)
	O:zero()
	for i = 1, M do
	    for j = i+1, M do
	        O[i][j] = States[i]:dot(States[j])/N
	    end
	end
end

--file = torch.DiskFile('/home/sagun/SpinGlass/overlap_' .. N .. '_' .. M .. '.asc', 'w')
--file:writeObject(O)
--file:close()
