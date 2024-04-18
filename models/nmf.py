import jax
import numpy as np
import jax.numpy as jnp


@jax.jit
# mse loss for batch
def loss(U_batch, S_batch, ratings_batch, lambda_reg):
    pred_ratings_batch = jnp.sum(jnp.multiply(U_batch, S_batch), axis=1)
    squared_diff = (pred_ratings_batch - ratings_batch) ** 2
    l2_reg = lambda_reg * (jnp.sum(jnp.abs(U_batch)) + jnp.sum(jnp.abs(S_batch)))
    return jnp.mean(squared_diff) + l2_reg
loss_grad_u = jax.jit( jax.value_and_grad(loss, argnums=(0, )) )
loss_grad_s = jax.jit( jax.value_and_grad(loss, argnums=(1, )) )


def ce(u, s, levels, rating):
    pred_rating = jnp.array([1 if i == jnp.round(jnp.dot(u, s)) - 1 else 0 for i in range(levels)])
    true_rating = jnp.array([1 if i == rating - 1 else 0 for i in range(levels)])
    ce = - true_rating * jnp.log(pred_rating)
    return jnp.sum(ce)


# Non-negative Matrix Factorization (NMF) implementation using numpy and jax
class nmf():
    def __init__(self, n_users, n_songs, n_factors):
        self.U = np.random.normal(0, 1, size=(n_users, n_factors)) / np.sqrt(n_factors)
        self.S = np.random.normal(0, 1, size=(n_songs, n_factors)) / np.sqrt(n_factors)
    
    def train_batch(self, epoch_i, ratings, indices, learning_rate, lambda_reg, batch_size, mode='TRAIN'):
        n_batches = indices[0].shape[0] // batch_size
        losses = []
        for batch in range(n_batches):
            user_indices = indices[0][batch * batch_size : (batch + 1) * batch_size]
            song_indices = indices[1][batch * batch_size : (batch + 1) * batch_size]
            U_batch, S_batch = self.U[user_indices], self.S[song_indices]
            ratings_batch = np.array([ratings[user_indices[i], song_indices[i]] for i in range(batch_size)])
            loss_u, grads_u = loss_grad_u(U_batch, S_batch, ratings_batch, lambda_reg)
            loss_s, grads_s = loss_grad_s(U_batch, S_batch, ratings_batch, lambda_reg)
            if mode == 'TRAIN':
                # update param (make U and S non-negative)
                self.U[user_indices] = np.clip(self.U[user_indices] - learning_rate * np.clip(grads_u[0], -1, 1), 0, None)   
                self.S[song_indices] = np.clip(self.S[song_indices] - learning_rate * np.clip(grads_s[0], -1, 1), 0, None)
            losses.append(loss_s)
        return np.mean(losses)

    def train(self, train_data, val_data, n_epochs, learning_rate, lambda_reg, batch_size):
        # initialize loss, load data
        train_losses, val_losses = [], []
        train_ratings, train_indices = train_data[0], train_data[1]
        val_ratings, val_indices = val_data[0], val_data[1]
        # training
        for i in range(n_epochs):
            train_loss = self.train_batch(i, train_ratings, train_indices, learning_rate, lambda_reg, batch_size, mode='TRAIN')
            val_loss = self.train_batch(i, val_ratings, val_indices, learning_rate, lambda_reg, batch_size, mode='VAL')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if i % 1 == 0:
                print(f'Epoch {i} | training loss = {train_loss:.4f} | validation loss = {val_loss:.4f}')
        return train_losses, val_losses

    def predict(self, test_data):
        ratings_test, indices = test_data[0], test_data[1]
        U_test, S_test = self.U[indices[0]], self.S[indices[1]]
        ratings_test = np.array([ratings_test[indices[0][i], indices[1][i]] for i in range(len(indices[0]))])
        pred_ratings_test = jnp.sum(jnp.multiply(U_test, S_test), axis=1)
        return pred_ratings_test, ratings_test

            
        
