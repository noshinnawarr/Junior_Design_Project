n_train_images = 1000
train_images_folder = "data/train"
test_images_folder = "data/test"

import os
from falconcv.ds import Coco
from falconcv.util import FileUtil, ImageUtil, VIUtil, ColorUtil

def download_data(labels_map, color_palette, n_images, batch_size, split, task, output_folder):
    try:
        # creating dataset
        dataset = Coco(v=2017)
        dataset.setup(split=split, task=task)
        os.makedirs(output_folder, exist_ok=True)
        FileUtil.clear_folder(output_folder)
        for batch_images in dataset.fetch(
                n=n_images,
                labels=list(labels_map.keys()),
                batch_size=batch_size):
            for img in batch_images:
                img.export(output_folder, labels_map, color_palette)
    except Exception as ex:
        print(f"Error downloading dataset: {ex}")

labels_map = {
    "airplane": 1,
    "train": 2
}
color_palette = ColorUtil.color_palette(n=len(labels_map))

# download train images
download_data(labels_map=labels_map, color_palette=color_palette, n_images=n_train_images,
              batch_size=500, split="train", task="segmentation", output_folder=train_images_folder)



from falconcv.models.tf import ModelZoo

ModelZoo.print_available_models(arch="mask")

"""## 3.2. Entrenar modelo"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from falconcv.models import ModelBuilder

model_folder = "model/tf"
model = "mask_rcnn_inception_v2_coco"

def train_and_freeze_model(model_name, images_folder, out_folder, labels_map, epochs=5000):
    try:
        config = {
            "model": model_name,
            "images_folder": images_folder,
            "output_folder": out_folder,
            "labels_map": labels_map,
        }
        with ModelBuilder.build(config=config) as model:
            model.train(epochs=epochs, val_split=0.3, clear_folder=True)
            model.freeze(epochs)
    except Exception as ex:
        raise Exception(f"Error training the model {ex}") from ex

train_and_freeze_model(model, train_images_folder, model_folder, labels_map)



model = "mask_rcnn_inception_v2_coco"
frozen_model_file = os.path.join(model_folder, model, "export/frozen_inference_graph.pb")
labels_map_file = os.path.join(model_folder, model, "label_map.pbtxt")

def make_predictions(frozen_model, labels_map_file, image, threshold):
    # load freeze model
    with ModelBuilder.build(frozen_model, labels_map_file) as model:
        image, predictions = model.predict(image, threshold=threshold)
        fig = VIUtil.imshow(image, predictions)

from glob import glob

for image in glob(os.path.join(test_images_folder, "*")):
    make_predictions(frozen_model_file, labels_map_file, image, threshold=0.7)

