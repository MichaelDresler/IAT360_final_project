from mmaction.apis import inference_recognizer, init_recognizer
from mmengine import Config
from operator import itemgetter

# Choose to use a config and initialize the recognizer
config = 'configs/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb.py'
config = Config.fromfile(config)
# Setup a checkpoint file to load
checkpoint = 'checkpoints/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb_20220906-23cff032.pth'
# Initialize the recognizer
model = init_recognizer(config, checkpoint, device='cuda:0')

video = 'demo/dennis.mp4'
label = 'tools/data/kinetics/label_map_k400.txt'
results = inference_recognizer(model, video)
squat_percent = 0;

pred_scores = results.pred_score.tolist()
score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
top5_label = score_sorted[:5]

labels = open(label).readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in top5_label]

print("Detecting Squat")

for result in results:
    if result[0] == "squat" and result[1] > 0.9:
        squat = result[1]
        squat_percent = round(squat * 100, 2)
        

if squat_percent > 0.9:
    print(f"This video contains a squat with a probability of {squat_percent}%")   
else:
    print("this video does not contain a squat")

    

# for result in results:
#     print(f'{result[0]}: ', result[1])

    