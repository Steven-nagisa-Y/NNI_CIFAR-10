# coding=utf-8
import numpy

N, D_in, H, D_out = 64, 1000, 100, 10

x = numpy.random.rand(N, D_in)
y = numpy.random.rand(N, D_out)

w1 = numpy.random.randn(D_in, H)
w2 = numpy.random.randn(H, D_out)

learning_rate = 1e-6

for t in range(10000):
    h = x.dot(w1)
    h_relu = numpy.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    loss = numpy.square(y_pred - y).sum()
    print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2