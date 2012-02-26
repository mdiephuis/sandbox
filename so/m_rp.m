% 
%	m_rp - Generate randomprojection LxN matrix W, which consists of a set of 
%	approximately orthonormal basis vectors, where all elements are generated as
%	as Wi[j] ~ N(0; 1/N) and, 1 <= i <= N,  1 <= j <= L.
%	As such, it behaves as an approximate orthoprojector. 
%
%	Usage:
%					W = m_rp(L, N)
%
%	Arguments:
%					L:		Number of rows of W, or the number of dimensions to project to
%					N:		Number of cols of W, or input dimension
%	
% Returns:
%					W:		LxN randomprojection matrix
%
%
%	Authors:
%		Tarras Holotyak, Maurits Diephuis 
%
%	References:
%
%
%	Date:
%		7/28/2011
%
function W =  m_rp(L, N)

if ( L >= N )
   A = orth(randn(L, N)/sqrt(L));
else
   A = (orth((randn(L, N)/sqrt(L))'))'; 
end
  
B = A - ones(L, 1)*mean(A, 1);
C = B - ones(L, 1)*mean(B, 1);

W = C - ones(L, 1)*mean(C, 1);

W = orth(W);