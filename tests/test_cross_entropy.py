import unittest

import jax
import jax.numpy as jnp
import numpy as np

from training import cross_entropy


class CrossEntropyTest(unittest.TestCase):
    def test_fixed_implementations_match_standard_values(self):
        logits = jnp.asarray(
            [[1.25, -0.5, 2.0], [-3.0, 0.25, 0.75]],
            dtype=jnp.float32,
        )
        targets = jnp.asarray([2, 0])

        np.testing.assert_allclose(
            cross_entropy.fixed_log_softmax(logits),
            jax.nn.log_softmax(logits, axis=-1),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            cross_entropy.fixed_cross_entropy(logits, targets),
            -jax.nn.log_softmax(logits, axis=-1)[jnp.arange(2), targets],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_fixed_cross_entropy_preserves_small_positive_loss(self):
        logits = jnp.asarray([[20.0, 0.0, -20.0]], dtype=jnp.float32)
        targets = jnp.asarray([0])

        loss = np.asarray(
            cross_entropy.fixed_cross_entropy(logits, targets)
        )[0]
        expected = np.log1p(np.exp(-20.0) + np.exp(-40.0))

        self.assertGreater(loss, 0.0)
        self.assertAlmostEqual(float(loss), float(expected), places=14)

    def test_fixed_cross_entropy_is_shift_invariant(self):
        logits = jnp.asarray(
            [[4.0, -2.0, 0.5], [1.0, 3.0, -5.0]],
            dtype=jnp.float32,
        )
        targets = jnp.asarray([1, 2])
        shifts = jnp.asarray([[1000.0], [-1000.0]], dtype=jnp.float32)

        np.testing.assert_allclose(
            cross_entropy.fixed_cross_entropy(logits, targets),
            cross_entropy.fixed_cross_entropy(logits + shifts, targets),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_fixed_cross_entropy_gradient_matches_closed_form(self):
        logits = jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float32)
        target = jnp.asarray(1)
        gradient = jax.grad(cross_entropy.fixed_cross_entropy)(logits, target)
        expected = jax.nn.softmax(logits)
        expected = expected.at[target].add(-1.0)

        np.testing.assert_allclose(
            gradient,
            expected,
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
