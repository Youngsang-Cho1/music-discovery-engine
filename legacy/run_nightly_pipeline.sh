#!/bin/bash
# Nightly Automated Audio Pipeline
# Loops through batches 1 to 29 (since batch 0 is already running/done).
# After downloading and extracting each 1,000 song batch, it safely appends the 
# features to the database and deletes the heavy audio files to save space.

echo "========================================="
echo "   NIGHTLY ML PIPELINE (AUTO-CLEANUP)   "
echo "========================================="

# Adjust the start and end batch numbers depending on where you left off.
# Total dataset has ~30 batches (0 to 29).
START_BATCH=1
END_BATCH=29

for batch in $(seq $START_BATCH $END_BATCH); do
    echo ""
    echo "========================================="
    echo " ▶ STARTING BATCH $batch"
    echo "========================================="
    
    # 1. Download Batch and sleep to avoid angry Google servers
    /opt/anaconda3/bin/python src/fetch_youtube_audio.py --batch $batch
    
    # Check if download crashed completely format
    if [ $? -ne 0 ]; then
        echo "⚠️ Error in download batch $batch. Skipping to next to maintain pipeline."
        continue
    fi

    # 2. Extract Features (Now appends safely to parquet!)
    /opt/anaconda3/bin/python src/audio_feature_extractor.py
    
    # 3. Cleanup disabled — keeping all audio files until pipeline is complete
    # rm -f data/audio_files/*.m4a
    # rm -f data/audio_files/*.mp3
    
    # 4. Anti-ban mechanism (Cool down the YouTube API limit)
    echo "🕒 Cooling down for 60 seconds before next batch..."
    sleep 60
done

echo ""
echo "========================================="
echo " 🎉 NIGHTLY PIPELINE COMPLETED! 🎉"
echo " Run pca.py -> clustering.py -> train_contrastive.py to finish."
echo "========================================="
