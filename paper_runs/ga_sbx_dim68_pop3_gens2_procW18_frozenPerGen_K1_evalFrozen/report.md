# Automated Documentation Report

Generated: 2025-10-29 22:38:07

## ga_sbx_dim68_pop3_gens2_procW18_frozenPerGen_K1_evalFrozen
**Config**:
```json
{
  "name": "ga_sbx_dim68_pop3_gens2_procW18_frozenPerGen_K1_evalFrozen",
  "train_scenarios": [],
  "eval_scenarios": [
    {
      "name": "random_repeatable_frozen_g555_m0",
      "num_asteroids": null,
      "asteroid_states": [
        {
          "position": [
            655.2961828694579,
            837.9597537429767
          ],
          "angle": 151.63358054740016,
          "speed": 133.37272404713815,
          "size": 2
        },
        {
          "position": [
            819.9716515501843,
            354.3515874438207
          ],
          "angle": 80.11887628492028,
          "speed": 104.82814766244235,
          "size": 2
        },
        {
          "position": [
            997.8243549359355,
            742.5150997386935
          ],
          "angle": 85.17469876936373,
          "speed": 163.05804190959938,
          "size": 2
        },
        {
          "position": [
            557.503849786393,
            175.42044019606183
          ],
          "angle": 160.02421951911958,
          "speed": 106.44041562039473,
          "size": 1
        },
        {
          "position": [
            604.8129322006907,
            392.41421430430614
          ],
          "angle": 8.835085834799834,
          "speed": 147.0145458586736,
          "size": 3
        },
        {
          "position": [
            5.654603724015184,
            378.5051646009553
          ],
          "angle": 289.8931643804675,
          "speed": 56.541877647439925,
          "size": 2
        },
        {
          "position": [
            813.8736437203506,
            3.929186155770359
          ],
          "angle": 274.97116562291944,
          "speed": 90.88235588280601,
          "size": 3
        },
        {
          "position": [
            59.96606881904163,
            207.99118698310767
          ],
          "angle": 254.4272472308883,
          "speed": 46.837733444018895,
          "size": 3
        },
        {
          "position": [
            992.0089258538586,
            226.30366882908655
          ],
          "angle": 89.72656486044266,
          "speed": 117.07659440461633,
          "size": 4
        },
        {
          "position": [
            757.1057387593057,
            648.3479246159676
          ],
          "angle": 154.01638542045566,
          "speed": 152.51965487627325,
          "size": 3
        },
        {
          "position": [
            32.83777391503349,
            341.49191484685986
          ],
          "angle": 296.5639210343373,
          "speed": 121.39767465860238,
          "size": 2
        },
        {
          "position": [
            681.2511927396234,
            376.71854924907245
          ],
          "angle": 348.3353952494805,
          "speed": 117.26320486334782,
          "size": 3
        }
      ],
      "ship_states": [
        {
          "position": [
            400,
            400
          ],
          "angle": 90,
          "lives": 3,
          "team": 1
        }
      ],
      "map_size": [
        1200,
        900
      ],
      "time_limit": 60.0,
      "ammo_limit_multiplier": 0.0,
      "stop_if_no_ammo": false
    }
  ],
  "dim": 68,
  "optimizer": "ga_sbx",
  "pop": 3,
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
- seed 0: best=227.0

**Evaluation per seed**:
- seed 0: fitness_mean=131.368±0.000, score_mean=nan±0.000, acc_mean=0.974±0.000, hits_mean=40.000±0.000, deaths_mean=3.000±0.000
