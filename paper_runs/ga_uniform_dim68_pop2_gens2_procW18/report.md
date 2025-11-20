# Automated Documentation Report

Generated: 2025-10-28 09:29:04

## ga_uniform_dim68_pop2_gens2_procW18
**Config**:
```json
{
  "name": "ga_uniform_dim68_pop2_gens2_procW18",
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
  "optimizer": "ga_uniform",
  "pop": 2,
  "gens": 2,
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

**Training (per-seed)**:
- seed 0: best=246.43589743589743
- seed 1: best=207.0
- seed 2: best=187.0

**Evaluation per seed**:
- seed 0: fitness_mean=246.436±0.000, score_mean=nan±nan, acc_mean=0.974±0.000, hits_mean=155.000±0.000, deaths_mean=3.000±0.000
- seed 1: fitness_mean=207.000±0.000, score_mean=nan±nan, acc_mean=1.000±0.000, hits_mean=113.000±0.000, deaths_mean=3.000±0.000
- seed 2: fitness_mean=187.000±0.000, score_mean=nan±nan, acc_mean=1.000±0.000, hits_mean=97.000±0.000, deaths_mean=5.000±0.000
