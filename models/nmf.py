import jax
import numpy as np
import jax.numpy as jnp


@jax.jit
# mse loss for batch with regulation
def loss(U_batch, S_batch, ratings_batch, lambda_reg):
    pred_ratings_batch = jnp.sum(jnp.multiply(U_batch, S_batch), axis=1)
    squared_diff = (pred_ratings_batch - ratings_batch) ** 2
    l2_reg = lambda_reg * (jnp.sum(jnp.abs(U_batch)) + jnp.sum(jnp.abs(S_batch)))
    return jnp.mean(squared_diff) + l2_reg
compute_grad_u = jax.jit( jax.grad(loss, argnums=(0, )) )
compute_grad_s = jax.jit( jax.grad(loss, argnums=(1, )) )

@jax.jit
# mse loss for batch without regulation
def loss(U_batch, S_batch, ratings_batch):
    pred_ratings_batch = jnp.sum(jnp.multiply(U_batch, S_batch), axis=1)
    squared_diff = (pred_ratings_batch - ratings_batch) ** 2
    return jnp.mean(squared_diff)


# Non-negative Matrix Factorization (NMF) implementation using numpy and jax
class nmf():
    def __init__(self, n_users, n_songs, n_factors):
        self.U = np.random.uniform(0, 1, size=(n_users, n_factors)) # / np.sqrt(n_factors)
        self.S = np.random.uniform(0, 1, size=(n_songs, n_factors)) # / np.sqrt(n_factors)
    
    def train_batch(self, ratings, indices, learning_rate, lambda_reg, batch_size, mode='TRAIN'):
        n_batches = indices[0].shape[0] // batch_size
        losses = []
        for batch in range(n_batches):
            user_indices = indices[0][batch * batch_size : (batch + 1) * batch_size]
            song_indices = indices[1][batch * batch_size : (batch + 1) * batch_size]
            U_batch, S_batch = self.U[user_indices], self.S[song_indices]
            ratings_batch = np.array([ratings[user_indices[i], song_indices[i]] for i in range(batch_size)]) 
            if mode == 'TRAIN':
                # update param (make U and S non-negative)
                grad_u, grad_s = compute_grad_u(U_batch, S_batch, ratings_batch, lambda_reg), compute_grad_s(U_batch, S_batch, ratings_batch, lambda_reg)
                self.U[user_indices] = np.clip(self.U[user_indices] - learning_rate * np.clip(grad_u[0], -1, 1), 0, None)
                self.S[song_indices] = np.clip(self.S[song_indices] - learning_rate * np.clip(grad_s[0], -1, 1), 0, None)
            losses.append(loss(U_batch, S_batch, ratings_batch))
        return np.mean(losses)

    def train(self, train_data, val_data, n_epochs, learning_rate, lambda_reg, batch_size):
        # initialize loss, load data
        train_losses, val_losses = [], []
        train_ratings, train_indices = train_data[0], train_data[1]
        val_ratings, val_indices = val_data[0], val_data[1]
        # training
        for i in range(n_epochs):
            train_loss = self.train_batch(train_ratings, train_indices, learning_rate, lambda_reg, batch_size, mode='TRAIN')
            val_loss = self.train_batch(val_ratings, val_indices, learning_rate, lambda_reg, batch_size, mode='VAL')
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

            
        
