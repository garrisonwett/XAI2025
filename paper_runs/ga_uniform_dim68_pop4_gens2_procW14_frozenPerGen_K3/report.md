# Automated Documentation Report

Generated: 2025-10-29 20:57:58

## ga_uniform_dim68_pop4_gens2_procW14_frozenPerGen_K3
**Config**:
```json
{
  "name": "ga_uniform_dim68_pop4_gens2_procW14_frozenPerGen_K3",
  "train_scenarios": [],
  "eval_scenarios": [
    {
      "name": "training2",
      "num_asteroids": null,
      "asteroid_states": [
        {
          "position": [
            100,
            100
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            200
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            300
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            400
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            500
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            600
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        },
        {
          "position": [
            100,
            700
          ],
          "angle": 0,
          "speed": 100,
          "size": 4
        }
      ],
      "ship_states": [
        {
          "position": [
            400,
            400
          ],
          "angle": 90,
          "lives": 5,
          "team": 1
        }
      ],
      "map_size": [
        1000,
        800
      ],
      "time_limit": 60.0,
      "ammo_limit_multiplier": 0.0,
      "stop_if_no_ammo": false
    }
  ],
  "dim": 68,
  "optimizer": "ga_uniform",
  "pop": 4,
  "gens": 2,
  "seeds": 1,
  "episodes": 3,
  "workers": 14,
  "eval_timeout": null,
  "train_max_hours": 1,
  "frozen_random_per_gen": true,
  "frozen_maps_per_gen": 3,
  "frozen_num_asteroids": 12,
  "frozen_map_w": 1200,
  "frozen_map_h": 900
}
```

**Training per seed**:
- seed 0: best=191.0

**Evaluation per seed**:
- seed 0: fitness_mean=249.000±0.000, score_mean=nan±nan, acc_mean=1.000±0.000, hits_mean=155.000±0.000, deaths_mean=3.000±0.000
