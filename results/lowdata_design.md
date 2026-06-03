# Low-Data Regime Design

- unified: `data/sft/unified.jsonl`
- counterfactual judged: `data/counterfactual/cf_judged.json`
- n: [300, 500, 1000]
- sample seeds: [42]
- train seeds: [42, 43, 44]
- subject stratified: True

- `komed_low_c3_n300_r42` -> `data\sft\train_low_c3_n300_r42.jsonl`
  - C3: n=300, mean_tokens=185.6, subjects={'dentist': 24, 'doctor': 167, 'nurse': 52, 'pharm': 57}, gap_mean=4.000, c2_mean=4.410
- `komed_low_c2_n300_r42` -> `data\sft\train_low_c2_n300_r42.jsonl`
  - C2: n=300, mean_tokens=157.9, subjects={'dentist': 24, 'doctor': 167, 'nurse': 52, 'pharm': 57}, gap_mean=3.633, c2_mean=4.799
- `komed_low_crand_n300_r42` -> `data\sft\train_low_crand_n300_r42.jsonl`
  - C-rand: n=300, mean_tokens=193.1, subjects={'pharm': 57, 'doctor': 167, 'dentist': 24, 'nurse': 52}, gap_mean=3.323, c2_mean=4.340

- `komed_low_c3_n500_r42` -> `data\sft\train_low_c3_n500_r42.jsonl`
  - C3: n=500, mean_tokens=193.8, subjects={'dentist': 41, 'doctor': 277, 'nurse': 87, 'pharm': 95}, gap_mean=4.000, c2_mean=4.417
- `komed_low_c2_n500_r42` -> `data\sft\train_low_c2_n500_r42.jsonl`
  - C2: n=500, mean_tokens=156.1, subjects={'dentist': 41, 'doctor': 277, 'nurse': 87, 'pharm': 95}, gap_mean=3.534, c2_mean=4.746
- `komed_low_crand_n500_r42` -> `data\sft\train_low_crand_n500_r42.jsonl`
  - C-rand: n=500, mean_tokens=206.2, subjects={'doctor': 277, 'pharm': 95, 'nurse': 87, 'dentist': 41}, gap_mean=3.336, c2_mean=4.324

- `komed_low_c3_n1000_r42` -> `data\sft\train_low_c3_n1000_r42.jsonl`
  - C3: n=1000, mean_tokens=192.7, subjects={'dentist': 81, 'doctor': 555, 'nurse': 174, 'pharm': 190}, gap_mean=4.000, c2_mean=4.394
- `komed_low_c2_n1000_r42` -> `data\sft\train_low_c2_n1000_r42.jsonl`
  - C2: n=1000, mean_tokens=152.4, subjects={'dentist': 81, 'doctor': 555, 'nurse': 174, 'pharm': 190}, gap_mean=3.516, c2_mean=4.706
- `komed_low_crand_n1000_r42` -> `data\sft\train_low_crand_n1000_r42.jsonl`
  - C-rand: n=1000, mean_tokens=213.9, subjects={'doctor': 555, 'pharm': 190, 'dentist': 81, 'nurse': 174}, gap_mean=3.328, c2_mean=4.293
