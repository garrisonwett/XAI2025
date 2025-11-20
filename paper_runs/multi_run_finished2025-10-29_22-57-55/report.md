# Automated Documentation Report

Generated: 2025-10-29 22:57:43

## ga_sbx::ga_sbx
**Config**:
```json
{
  "name": "ga_sbx_dim68_pop4_gens2_procW18_frozenPerGen_K1",
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
  "optimizer": "ga_sbx",
  "pop": 4,
  "gens": 2,
  "seeds": 1,
  "episodes": 1,
  "workers": 18,
  "eval_timeout": null,
  "train_max_hours": 1,
  "frozen_random_per_gen": true,
  "frozen_maps_per_gen": 1,
  "frozen_num_asteroids": 12,
  "frozen_map_w": 1200,
  "frozen_map_h": 900
}
```

**Training per seed**:
- seed 0: best=225.2248062015504

**Evaluation per seed**:
- seed 0: fitness_mean=219.206±0.000, score_mean=nan±0.000, acc_mean=0.992±0.000, hits_mean=130.000±0.000, deaths_mean=5.000±0.000
