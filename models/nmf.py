import jax
import numpy as np
import jax.numpy as jnp


@jax.jit
# mse loss for batch
def loss(U_batch, S_batch, ratings_batch, lambda_reg):
    pred_ratings_batch = jnp.sum(jnp.multiply(U_batch, S_batch), axis=1)
    squared_diff = (pred_ratings_batch - ratings_batch) ** 2
    l2_reg = lambda_reg * (jnp.sum(U_batch ** 2) + jnp.sum(S_batch ** 2))
    
    return jnp.mean(squared_diff)
loss_grad = jax.jit( jax.value_and_grad(loss, argnums=(0, 1)) )


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
    
    def train_batch(self, ratings, learning_rate, lambda_reg, n_batches, batch_size, mode='TRAIN'):
        n_users, n_songs = self.U.shape[0], self.S.shape[0]
        for batch in range(n_batches):
            # randomly select batch samples
            user_indices = np.random.choice(n_users, batch_size, replace=False)
            song_indices = np.random.choice(n_songs, batch_size, replace=False)
            U_batch, S_batch = self.U[user_indices], self.S[song_indices]
            ratings_batch = np.array([ratings[user_indices[i]][song_indices[i]] for i in range(batch_size)])
            loss, grads = loss_grad(U_batch, S_batch, ratings_batch, lambda_reg)
            if mode == 'TRAIN':
                # update param (make U and S non-negative)
                self.U[user_indices] = np.clip(self.U[user_indices] - learning_rate * np.clip(grads[0], -1, 1), 0, None)
                self.S[song_indices] = np.clip(self.S[song_indices] - learning_rate * np.clip(grads[1], -1, 1), 0, None)
            return loss

    def train(self, ratings_train, ratings_val, n_epochs, learning_rate, lambda_reg, n_batches, batch_size):
        train_losses, val_losses = [], []
        for i in range(n_epochs):
            train_loss = self.train_batch(ratings_train, learning_rate, lambda_reg, n_batches, batch_size, mode='TRAIN')
            val_loss = self.train_batch(ratings_val, learning_rate, lambda_reg, n_batches, batch_size, mode='VAL')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if i % 10 == 0:
                print(f'Epoch {i} | training loss = {train_loss:.4f} | validation loss = {val_loss:.4f}')
        return train_losses, val_losses

            
            
        
