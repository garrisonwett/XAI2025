# Automated Documentation Report

Generated: 2025-10-29 09:34:45

## cma_dim68_pop40_gens40_procW18
**Config**:
```json
{
  "name": "cma_dim68_pop40_gens40_procW18",
  "train_scenarios": [
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
  "optimizer": "cma",
  "pop": 40,
  "gens": 40,
  "seeds": 3,
  "episodes": 3,
  "workers": 18,
  "eval_timeout": null,
  "train_max_hours": 1,
  "frozen_random_per_gen": false,
  "frozen_maps_per_gen": 3,
  "frozen_num_asteroids": 12,
  "frozen_map_w": 1200,
  "frozen_map_h": 900
}
```

**Training per seed**:
- seed 0: best=181.80219780219778
- seed 1: best=190.9795918367347
- seed 2: best=190.0

**Evaluation per seed**:
- seed 0: fitness_mean=181.802±0.000, score_mean=nan±nan, acc_mean=0.978±0.000, hits_mean=94.000±0.000, deaths_mean=5.000±0.000
- seed 1: fitness_mean=190.980±0.000, score_mean=nan±nan, acc_mean=0.990±0.000, hits_mean=102.000±0.000, deaths_mean=5.000±0.000
- seed 2: fitness_mean=190.000±0.000, score_mean=nan±nan, acc_mean=1.000±0.000, hits_mean=100.000±0.000, deaths_mean=5.000±0.000
