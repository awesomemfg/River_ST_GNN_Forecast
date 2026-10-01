"""Model definitions shared by the trainer (mike) and the evaluators (<SERVER>), so both build exactly the
same Keras variables in the same order.

STGNNQ
    The paper's ST-GNN, identical to the class in
    archive/model_lineage/stgnn_train_reforecast_2026.py:
    a shared GRU encodes each gauge's 72 h history, then a shared GRU cell decodes 96 steps, taking the
    known forcing and the graph message sum_u A[d,u] h_u at every step. Dense head -> 5 quantiles.

NodewiseLSTMQ
    The established LSTM baseline: one LSTM encoder and one LSTM decoder cell shared by all gauges, run
    gauge by gauge with NO information exchanged between gauges. The decoder receives the same forcing
    and, in place of the graph message, the gauge's own hidden state, exactly as in
    archive/model_lineage/lstm_train_nodewise.py.
"""
import tensorflow as tf


class STGNNQ(tf.keras.Model):
    def __init__(self, adjacency, n_nodes, in_w, lb_w, in_ch, f_ch, hidden, n_quantiles):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_w = in_w
        self.lb_w = lb_w
        self.in_ch = in_ch
        self.f_ch = f_ch
        self.hidden = hidden
        self.n_quantiles = n_quantiles
        self.A = tf.constant(adjacency)
        self.enc = tf.keras.layers.GRU(hidden)
        self.cell = tf.keras.layers.GRUCell(hidden)
        self.out = tf.keras.layers.Dense(n_quantiles)

    def call(self, inp, training=False):
        h, forcing, z0 = inp
        batch = tf.shape(h)[0]
        n = self.n_nodes
        seq = tf.reshape(tf.transpose(h, [0, 2, 1, 3]), [batch * n, self.in_w, self.in_ch])
        hs = tf.reshape(self.enc(seq), [batch, n, self.hidden])
        outs = tf.TensorArray(tf.float32, size=self.lb_w)
        for t in range(self.lb_w):
            msg = tf.einsum("du,buh->bdh", self.A, hs)
            inp_t = tf.concat([forcing[:, t], msg], -1)
            h2, _ = self.cell(tf.reshape(inp_t, [batch * n, self.f_ch + self.hidden]),
                              [tf.reshape(hs, [batch * n, self.hidden])])
            hs = tf.reshape(h2, [batch, n, self.hidden])
            outs = outs.write(t, tf.reshape(self.out(hs), [batch, n, self.n_quantiles]))
        return tf.transpose(outs.stack(), [1, 0, 2, 3])


class NodewiseLSTMQ(tf.keras.Model):
    def __init__(self, n_nodes, in_w, lb_w, in_ch, f_ch, hidden, n_quantiles):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_w = in_w
        self.lb_w = lb_w
        self.in_ch = in_ch
        self.f_ch = f_ch
        self.hidden = hidden
        self.n_quantiles = n_quantiles
        self.enc = tf.keras.layers.LSTM(hidden, return_state=True)
        self.cell = tf.keras.layers.LSTMCell(hidden)
        self.out = tf.keras.layers.Dense(n_quantiles)

    def call(self, inp, training=False):
        h, forcing, z0 = inp
        batch = tf.shape(h)[0]
        n = self.n_nodes
        seq = tf.reshape(tf.transpose(h, [0, 2, 1, 3]), [batch * n, self.in_w, self.in_ch])
        _, hidden_state, cell_state = self.enc(seq)
        outs = tf.TensorArray(tf.float32, size=self.lb_w)
        for t in range(self.lb_w):
            self_message = tf.reshape(hidden_state, [batch, n, self.hidden])
            inp_t = tf.concat([forcing[:, t], self_message], -1)
            decoded, state = self.cell(tf.reshape(inp_t, [batch * n, self.f_ch + self.hidden]),
                                       [hidden_state, cell_state])
            hidden_state = state[0]
            cell_state = state[1]
            node_hidden = tf.reshape(decoded, [batch, n, self.hidden])
            outs = outs.write(t, tf.reshape(self.out(node_hidden), [batch, n, self.n_quantiles]))
        return tf.transpose(outs.stack(), [1, 0, 2, 3])


def build_model(kind, adjacency, n_nodes, in_w, lb_w, in_ch, f_ch, hidden, n_quantiles):
    if kind == "stgnn":
        return STGNNQ(adjacency, n_nodes, in_w, lb_w, in_ch, f_ch, hidden, n_quantiles)
    if kind == "lstm":
        return NodewiseLSTMQ(n_nodes, in_w, lb_w, in_ch, f_ch, hidden, n_quantiles)
    raise ValueError("unknown model kind: " + str(kind))
