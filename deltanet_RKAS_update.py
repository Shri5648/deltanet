import torch
import torch.nn as nn

def chunk_batched_delta_rule_forward(Q,K,V,beta,C):
    """
        Q,K and V are of shape (B,L,d) where B is the batch, L the sequence length and d the embeding's dimension
        beta is of shape (B,L,1)
        C is the size of the chunk, it should divide L
    """
    B,L,d = Q.shape
    Q, K, V= map(lambda x: x.reshape(B,-1,C,d), [Q,K,V])
    beta = beta.reshape(B,-1,C)
    K_beta = K*beta.unsqueeze(-1)
    V_beta = V*beta.unsqueeze(-1)

    mask = torch.triu(torch.ones(C,C), diagonal=0).bool()

    K_t = torch.transpose(K,2,3)
    T = -(K_beta[:] @ K_t[:]).masked_fill(mask,0)
   
   #forward substitution 
    for k in range(L//C):
        for i in range(1,C):
            T_new = T.clone()
            T_new[:,k,i,:i] = T[:,k,i,:i] + (T[:,k,i,:,None]*T[:,k,:,:i]).sum(-2)
            T = T_new
        T[:,k] = T[:,k] + torch.eye(C)

    W = T @ K_beta
    U = T @ V_beta

    S = torch.zeros((B,d,d))
    O = torch.empty_like(V)
    mask = torch.triu(torch.ones(C,C), diagonal=1).bool()

    for i in range(L//C):
        q_i, k_i, w_i = Q[:,i], K[:,i], W[:,i]
        u_i = U[:,i]-w_i@S
        o_inter = q_i @ S
        A_i = (q_i @ k_i.transpose(1,2)).masked_fill(mask,0)
        o_intra = A_i @ u_i
        S = S + k_i.transpose(1,2)@u_i
        O[:,i] = o_intra + o_inter
        
    return O.reshape(B,L,d)

def delta_rule_recurrent_step(q_t, k_t, v_t, S_prev, r_prev=None, A_gram_col=None):
    """
    Updated for RKAS logic:
    x^{k+1} = x^k - alpha_k * A_ik
    r^{k+1} = r^k - alpha_k * A * A_ik
    """
    # In this context: 
    # S_prev is x^k (the state matrix)
    # k_t is A_ik (the current key/row)
    # A_gram_col is AA_ik^T (the column of the Gram matrix)
    # r_prev is the residual vector r^k
    
    if A_gram_col is None:
        # Fallback if Gram info isn't provided: assume A*A^T is identity-like
        A_gram_col = torch.outer(k_t, k_t) #..................
        
    if r_prev is None:
        # Initial residual r^0 = Ax^0 - b = -v_t (since x^0=0)
        r_prev = -v_t

    # Step 2: Compute alpha_k = <AA_ik^T, r^k> / ||AA_ik^T||^2
    # We treat each dimension of the residual matrix separately
    num = torch.sum(A_gram_col * r_prev)
    den = torch.sum(A_gram_col**2) + 1e-6
    alpha_k = num / den

    # Step 3: Update state (x^{k+1} = x^k - alpha_k * A_ik^T)
    # We use an outer product here because S is a matrix mapping k -> v
    # This corresponds to: S_new = S_prev - alpha_k * outer(1, k_t)
    # Adjusted for the multi-dimensional case:
    S_new = S_prev - alpha_k * torch.outer(torch.ones_like(v_t), k_t)
    
    # Step 3: Update residual (r^{k+1} = r^k - alpha_k * AA_ik^T)
    r_new = r_prev - alpha_k * A_gram_col
    
    # Compute output
    o_t = S_new @ q_t
    
    return o_t, S_new, r_new

class DeltaBlock(nn.Module):
    def __init__(self,d,expand=1, neg_eigen=False):
        """
            d is the dimension of the input
            d*expand is the size of the hidden state
            neg_eigen if true allow the model to have negative eigen value. It was not on the original paper but on another: https://arxiv.org/abs/2411.12537.
        """
        super(DeltaBlock,self).__init__()
        self.d = d
        self.expand = expand
        self.Wq = nn.Linear(d,d*expand)
        self.Wk = nn.Linear(d,d*expand)
        self.Wv = nn.Linear(d,d*expand)

        self.proj_out = nn.Linear(d*expand,d)

        self.beta = nn.Linear(d,1)
        self.sigma = nn.Sigmoid()
        self.alpha = 2 if neg_eigen else 1

    def forward(self, X, chunk=1):
        if chunk == 1:
            _, chunk, _ = X.shape        

        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        
        # --- RKAS Adaptive Step Size Logic ---
        # 1. Compute Gram Matrix Row info: G = K @ K.transpose(-1, -2)
        # 2. AA_ik^T corresponds to the columns of the key Gram matrix
        # For simplicity in the batched version, we use the norm-based alpha:
        
        k_norm_sq = torch.sum(K**2, dim=-1, keepdim=True)
        # In RKAS, alpha involves the residual. For the parallel chunked form, 
        # we often use the normalized version which is the foundation of RK:
        beta_projection = 1.0 / (k_norm_sq + 1e-6) 
        
        # If you have a specific 'b' or 'target' for the residual, 
        # you would calculate alpha_k = (dot(AA_i, r)) / ||AA_i||^2 here.
        
        return self.proj_out(chunk_batched_delta_rule_forward(Q, K, V, beta_projection, chunk))

    def step(self,X,S=None):
        """
            this is the parallel form of deltanet
            input:
                X vector of shape d
                S state of shape (d,d), if not provided, the model will initialize it with zeros(0,0)
            output:
                Y vector of shape d
                S new state of shape (d,d)
        """
        if S==None:
            S = torch.zeros(self.d*self.expand,self.d*self.expand)
        y,S = delta_rule_recurrent_step(self.Wq(X),self.Wk(X),self.Wv(X)/self.alpha,S)
        return self.proj_out(y), S
