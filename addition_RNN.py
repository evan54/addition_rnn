import numpy as np
import matplotlib.pyplot as plt

sigmoid = lambda x : 1 / (1 + np.exp(-x))

class RNN(object):

    def __init__(self, n_bits, dim_hidden, factor=1):
        np.random.seed(0)
        self.dim_input = 2
        self.dim_hidden = dim_hidden
        self.n_bits = n_bits
        self.dims = np.array([[1, 1], [1, self.dim_hidden], 
                [self.dim_hidden, 1], [self.dim_hidden, 2], 
                [self.dim_hidden, self.dim_hidden]])

        # Initialise NN grads of matrices / params
        self.args_original = np.random.randn(self.dims.prod(1).sum(), 1)
        self.b_y, self.W_yh, self.b_h, self.W_hx, self.W_hh = \
            self._unpack_args(self.args_original.copy(), self.dims)

        self.xbs, self.ybs = self.generate_sample(100)
        
    def _pack_args(self, *args):
        '''
        b_y_grad, W_hy_grad, b_h_grad, W_xh_grad, W_hh_grad
        '''
        out_args = []
        for arg in args:
            out_args.extend(arg.flatten().tolist())
        out_args = np.array(out_args)[None].T
        return out_args

    def _unpack_args(self, args, dims):
        '''
        returns b_y_grad, W_hy_grad, b_h_grad, W_xh_grad, W_hh_grad
        from a vector
        '''
        ii = 0
        out = []
        for dim in dims:
            h, w = dim
            out.append(args[ii:ii+h*w].reshape([h, w]))
            ii += h*w
        return out

    def _check_grad(self):
        args = self.args_original.copy()
        args_grad_man = np.empty_like(args)
        xb, yb = self.generate_sample()
        print('Calculating gradients manually...')
        # manual calc
        delta = 1e-5
        for i_arg, _ in enumerate(args):
            # + epsilon
            args[i_arg] += delta
            self.b_y, self.W_yh, self.b_h, self.W_hx, self.W_hh = \
                self._unpack_args(args.copy(), self.dims)
            _, _, err_plus = self.forward_pass(xb, yb)
            # - epsilon
            args[i_arg] -= 2*delta
            self.b_y, self.W_yh, self.b_h, self.W_hx, self.W_hh = \
                self._unpack_args(args.copy(), self.dims)
            _, _, err_minus = self.forward_pass(xb, yb)
            args_grad_man[i_arg] = (.5 * (err_plus**2 - err_minus**2).sum() /
                    (2 * delta))
            # restore arg
            args[i_arg] += delta
        
        print('Calculating gradients by backprop...')
        # forward pass
        self.b_y, self.W_yh, self.b_h, self.W_hx, self.W_hh = \
                self._unpack_args(self.args_original.copy(), self.dims)
        y_est, h, err = self.forward_pass(xb, yb)
        # backward pass
        grad_list = self.backward_pass(xb, yb, h, y_est, err)
        args_grad = self._pack_args(*grad_list)

        print('Did the two calculations match?', np.allclose(args_grad, args_grad_man))

        values = np.linspace(-1, 1) * 10
        err = np.empty_like(values)
        for i_val, value in enumerate(values):
            self.b_y, self.W_yh, self.b_h, self.W_hx, self.W_hh = \
                self._unpack_args(args + args_grad * value, self.dims)
            y_est, h, err0 = self.forward_pass(xb, yb)
            err[i_val] = np.sum(err0**2)
        plt.plot(values, err)

    def generate_sample(self, n_samples=1):
        x = np.random.randint(0, 2**(self.n_bits-1), size=[2, n_samples])
        y = x.sum(0)
        xb = np.unpackbits(np.uint8(x)[:,None,:], axis=1)[:, ::-1]
        yb = np.unpackbits(np.uint8(y)[None, :], axis=0)[::-1]
        return xb.squeeze(), yb.squeeze()

    def forward_step(self, x, h):
        assert x.shape == (2 , 1)
        assert h.shape == (self.dim_hidden, 1)
        v = self.W_hx @ x + self.W_hh @ h + self.b_h
        h = sigmoid(v)
        u = self.W_yh @ h + self.b_y
        y_est = sigmoid(u)
        return y_est, h

    def forward_pass(self, xb, yb):
        h = np.zeros([self.dim_hidden, self.n_bits+1])
        y_est = np.zeros(self.n_bits)
        for i_bit in range(self.n_bits):
            y_est[i_bit], h[:, [i_bit+1]] = self.forward_step(xb[:, [i_bit]], 
                    h[:, [i_bit]])
        err = y_est - yb
        return y_est, h, err
    
    def backward_step(self, x, y, y_est, h, h_next, dedh):
        assert x.shape == (2, 1)
        assert h.shape == (self.dim_hidden, 1)
        err = y_est - y
        b_y_grad = err * y_est * (1 - y_est)    # 1 x 1
        W_yh_grad = b_y_grad * h_next.T         # 1 x dim_hidden
        dedh += b_y_grad * self.W_yh.T          # dim_hidden x 1
        b_h_grad = dedh * h_next * (1 - h_next) # dim_hidden x 1
        W_hx_grad = b_h_grad * x.T              # dim_hidden x dim_input
        W_hh_grad = b_h_grad * h.T              # dim_hidden x dim_hidden
        dedh = self.W_hh.T @ b_h_grad           # dim_hidden x 1
        return b_y_grad, W_yh_grad, b_h_grad, W_hx_grad, W_hh_grad, dedh

    def backward_pass(self, xb, yb, h, y_est, err):
        dedh = np.zeros([self.dim_hidden, 1])
        b_y_grad, W_yh_grad, b_h_grad, W_hx_grad, W_hh_grad = \
                self._unpack_args(np.zeros_like(self.args_original), self.dims)
        for i_bit in range(self.n_bits)[::-1]:
            res = self.backward_step(xb[:, [i_bit]], 
                    yb[i_bit], y_est[i_bit],
                    h[:, [i_bit]], h[:, [i_bit+1]], dedh)
            b_y_grad += res[0]
            W_yh_grad += res[1]
            b_h_grad += res[2]
            W_hx_grad += res[3]
            W_hh_grad += res[4]
            dedh = res[5]
        return b_y_grad, W_yh_grad, b_h_grad, W_hx_grad, W_hh_grad
    
    def train(self, n_epochs, n_batch, progressBar=None, alpha=.01):
        n_epochs = int(n_epochs)
        np.random.seed(100)
        err = np.zeros(n_epochs)
        for epoch in range(n_epochs):
            if progressBar:
                progressBar.value = epoch
            b_y_grad = np.zeros_like(self.b_y)
            W_yh_grad = np.zeros_like(self.W_yh)
            b_h_grad = np.zeros_like(self.b_h)
            W_hx_grad = np.zeros_like(self.W_hx)
            W_hh_grad = np.zeros_like(self.W_hh)
            for i_sample in range(n_batch):
                xb, yb = self.generate_sample()
                y_est, h, err0 = self.forward_pass(xb, yb)
                (b_y_grad, W_yh_grad, b_h_grad, W_hx_grad,
                        W_hh_grad) = self.backward_pass(xb, yb, h, y_est, err)
                err[epoch] += np.sum(err0**2)
            self.b_y -= b_y_grad * alpha
            self.W_yh -= W_yh_grad * alpha
            self.b_h -= b_h_grad * alpha
            self.W_hx -= W_hx_grad * alpha
            self.W_hh -= W_hh_grad * alpha
        return err

    def predict(self, x1, x2):
        x = np.array([[x1], [x2]], dtype=np.uint8)
        y = np.array([x1 + x2], dtype=np.uint8)
        xb = np.unpackbits(x[:,None,:], axis=1)[:, ::-1]
        yb = np.unpackbits(y[None, :], axis=0)[::-1]
        xb = xb.squeeze()
        yb = yb.squeeze()
        y_est, h, err = self.forward_pass(xb, yb)
        y_est_b = y_est > 0.5
        return np.packbits(y_est_b[::-1])[0]


rnn = RNN(8, 16)
