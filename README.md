# Neural-Transfer-Realtime




## Training

python trainer.py \
--alpha 1 \
--beta 1 \
--image_dim 256 \
--batch_size 4 \
--steps_per_epoch 10 \
--epochs 10 \
--coco_dirpath ../Datasets/coco_unlabeled_2017 \
--save_weights_path ./models/a_muse_1_256_3.h5 \
--style_image_path ./images/style_images/a_muse.jpg

python trainer.py --alpha 1 --beta 1 --image_dim 256 --batch_size 4 --steps_per_epoch 10 --epochs 1 --coco_dirpath ../Datasets/coco_unlabeled_2017 --save_weights_path ./models/a_muse_test1.h5 --style_image_path ./images/style_images/a_muse.jpg


## Inference - Image

python image_inference.py \
--weights_path ./models/a_muse_test1.h5 \
--content_image_path ./images/content_images/european_building.jpg

python image_inference.py --weights_path ./models/a_muse_test1.h5 --content_image_path ./images/content_images/european_building.jpg


## Inference - Video

python video_inference.py \
--weights_path ./models/a_muse_test1.h5 \
--content_video_path ./images/content_videos/SUNP0025.AVI \
--combination_video_path ./images/combined_videos/a_muse_test1.AVI

python video_inference.py --weights_path ./models/a_muse_test1.h5 --content_video_path ./images/content_videos/SUNP0025.AVI --combination_video_path ./images/combined_videos/a_muse_test1.AVI