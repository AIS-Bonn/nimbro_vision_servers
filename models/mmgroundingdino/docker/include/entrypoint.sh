#!/bin/bash

SEARCH_DIR=/cache/mmgroundingdino

mkdir -p "$SEARCH_DIR"

# Define the list of URLs
urls=(  
  "https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth"
  "https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco/grounding_dino_swin-t_finetune_16xb2_1x_coco_20230921_152544-5f234b20.pth"
  "https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth"
  "https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco/grounding_dino_swin-b_finetune_16xb2_1x_coco_20230921_153201-f219e0c0.pth"
  "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"
  "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_all/grounding_dino_swin-b_pretrain_all-f9818a7c.pth"
  "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth"
  "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
  "https://huggingface.co/fushh7/LLMDet/resolve/main/tiny.pth"
  "https://huggingface.co/fushh7/LLMDet/resolve/main/base.pth"
  "https://huggingface.co/fushh7/LLMDet/resolve/main/large.pth"
)

# Loop through the URLs array
for url in "${urls[@]}"; do
  # Extract the filename from the URL
  filename=$(basename "$url")

  # Construct the full path for the file
  filepath="$SEARCH_DIR/$filename"

  # Check if the file already exists
  if [ ! -f "$filepath" ]; then
    # File doesn't exist, download it
    echo "Downloading $filename to $SEARCH_DIR..."
    wget -P "$SEARCH_DIR" "$url"
  else
    echo "$filename already exists in $SEARCH_DIR, skipping..."
  fi
done

bertpath="/cache/mmgroundingdino/transformers/mm_groundingdino"
mkdir -p "$bertpath"

# Check if the file does not exist
if [ ! -f "$bertpath/model.safetensors" ]; then
    # Download the zip file
    curl -J https://uni-bonn.sciebo.de/s/RdyPXYEW6oqropw/download -o "$SEARCH_DIR/shared.zip"
    # Unzip the downloaded file
    unzip "$SEARCH_DIR/shared.zip" -d "$SEARCH_DIR"
    # Remove the zip file
    rm "$SEARCH_DIR/shared.zip"
    # Move the required .pth file
    mv "$SEARCH_DIR/shared/bert_groundingdino.safetensors" "$bertpath/model.safetensors"
    # Move the other required files
    mv "$SEARCH_DIR/shared/vocab.txt" "$bertpath/vocab.txt"
    mv "$SEARCH_DIR/shared/tokenizer_config.json" "$bertpath/tokenizer_config.json"
    mv "$SEARCH_DIR/shared/tokenizer.json" "$bertpath/tokenizer.json"
    mv "$SEARCH_DIR/shared/special_tokens_map.json" "$bertpath/special_tokens_map.json"
    mv "$SEARCH_DIR/shared/config.json" "$bertpath/config.json"
    # Remove the directory now that we are done
    rm -r "$SEARCH_DIR/shared"
else
    echo "File $bertpath/model.safetensors already exists, skipping download and extraction."
fi

python /usr/local/bin/download_nltk_models.py


exec "$@"