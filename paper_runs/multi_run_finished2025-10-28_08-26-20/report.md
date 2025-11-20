# Automated Documentation Report

Generated: 2025-10-28 08:26:20

## ga_sbx::ga_sbx
**Config**:
```json
{
  "name": "ga_sbx_dim68_pop4_gens4_procW18",
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
  "optimizer": "ga_sbx",
  "pop": 4,
  "gens": 4,
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
- seed 0: best=251.125
- seed 1: best=233.22222222222223
- seed 2: best=270.87005649717514

**Evaluation per seed**:
- seed 0: fitness_mean=251.125±0.000, score_mean=nan±nan, acc_mean=0.981±0.000, hits_mean=161.000±0.000, deaths_mean=4.000±0.000
- seed 1: fitness_mean=233.222±0.000, score_mean=nan±nan, acc_mean=0.972±0.000, hits_mean=144.000±0.000, deaths_mean=4.000±0.000
- seed 2: fitness_mean=270.870±0.000, score_mean=nan±nan, acc_mean=0.989±0.000, hits_mean=178.000±0.000, deaths_mean=3.000±0.000
