# CELLO
This is the official project website for the paper [CELLO: Causal Evaluation of Large Vision-Language Models](https://arxiv.org/pdf/2406.19131).

## Causality in Vision-Language Context
<p align="center">
<img src="https://github.com/OpenCausaLab/CELLO/blob/e199a1268fc9bf3520a59aaf0b55aec3b9d12593/images/causal_relation.png" alt="" width="40%">
</p>

## Pipeline
<p align="center">
<img src="https://github.com/OpenCausaLab/CELLO/blob/e199a1268fc9bf3520a59aaf0b55aec3b9d12593/images/dataset.png" alt="" width="60%">
</p>

## Dataset
### Link
For images: Download from [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html).

For question, answers, etc: [CELLO Dataset](https://github.com/OpenCausaLab/CELLO/blob/5860767b2a213f8527a8ae3e42e0cf60546c8b66/data/cello_data.jsonl)

### Comparison
<p align="center">
<img src="https://github.com/OpenCausaLab/CELLO/blob/4aa79ef7f35bab76bfbb69ff7bbfb856cc3e4b36/images/comparisin.png" alt="" width="80%">
</p>

### Data Format
```JSON5
{
	"img_id": 2362590,
	"question": "If the beach were to suddenly erode, would the women still be able to ride the horses securely?",
	"graph_type": "confounding",
	"task_type": "sufficient_cause",
	"graph": {
		"nodes": [
			[3749557, {
				"obj_name": "beach",
				"obj_synsets": ["beach.n.01"],
				"boxes": [4, 194, 493, 174],
				"position": "right_down"
			}],
			[2037998, {
				"obj_name": "horse",
				"obj_synsets": ["horse.n.01"],
				"boxes": [
					[101, 170, 107, 110]
				],
				"position": "left_down"
			}],
			[1946903, {
				"obj_name": "women",
				"obj_synsets": ["woman.n.01"],
				"boxes": [
					[290, 126, 70, 113]
				],
				"position": "right_upper"
			}]
		],
		"edges": [
			[3749557, 2037998, {
				"relation": "ON"
			}],
			[3749557, 1946903, {
				"relation": "ON"
			}],
			[2037998, 1946903, {
				"relation": "riding"
			}]
		]
	},
	"objs": [3749557, 2037998, 1946903],
	"options": ["Yes", "No"],
	"answer_index": 1,
	"data_id": 5
}
```

### Statistics
<p align="center">
<img src="https://github.com/OpenCausaLab/CELLO/blob/4aa79ef7f35bab76bfbb69ff7bbfb856cc3e4b36/images/stat.png" alt="" width="40%">
</p>

## Evaluation
```
python evaluate_cello.py --model [MODEL_NAME]
```


## Citation
Please cite our paper if this repository inspires your work.
```bibtex
@article{chen2024cello,
  title={CELLO: Causal Evaluation of Large Vision-Language Models},
  author={Chen, Meiqi and Peng, Bo and Zhang, Yan and Lu, Chaochao},
  journal={arXiv preprint arXiv:2406.19131},
  year={2024}
}
```

## Contact 
- meiqichen@stu.pku.edu.cn
- peng_bo2019@sjtu.edu.cn
- zhyzhy001@pku.edu.cn
- luchaochao@pjlab.org.cn
