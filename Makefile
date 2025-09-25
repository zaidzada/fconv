
make-env:
	conda create -n fconv
	conda activate fconv
	pip install accelerate himalaya nilearn scipy scikit-learn spacy tqdm \
			    transformers voxelwise_tutorials gensim pandas matplotlib \
				seaborn torch torchaudio torchvision surfplot neuromaps \
				jupyter tqdm nltk statsmodels h5py netneurotools openpyxl natsort

download-atlas:
	# may not be needed. but need to run atlas.ipynb.
	wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_MD_HE197.nii
	wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_MD_HE197.txt
	wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_language_SN220.nii
	wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_language_SN220.txt
	wget https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_1000Parcels_Kong2022_17Networks_order_FSLMNI152_1mm.nii.gz

split-audio:
	python code/split_audio_clips.py


transcribe:
	# prerequisites:
	# git clone https://huggingface.co/guillaumekln/faster-whisper-large-v2 large-ct2
	# git lfs pull --include=model.bin
	# export HF_TOKEN =""

	# salloc --mem=4G --time=01:00:00 --gres=gpu:1
	whisperx \
			--model models/large-ct2 \
			--output_dir data/stimuli/whisperx \
			--output_format json \
			--task transcribe \
			--language en \
			--diarize \
			--min_speakers 2 --max_speakers 2 \
			--hf_token "${HF_TOKEN}" \
			--device cuda \
			data/stimuli/conv-*/audio/*condition-G*wav

process_trancsripts:
	python code/move_whisper_transcripts.py

llm_embeddings:
	sbatch --job-name=emb --mem=8G --time=00:05:00 --gres=gpu:1 code/slurm.sh -- \
		code/embeddings.py -m gpt2-2b --layer 24
	# python code/embeddings.py -m gpt2-2b --layer 0

generate_features:
	python code/featuregen.py spectral
	python code/featuregen.py articulatory
	python code/featuregen.py syntactic
	python code/featuregen.py bow

confound_regression:
	python code/clean.py -m default_task

black_llm:
	python code/black_encoding.py -m gpt2-xl --extract-only 

encoding:
	sbatch --job-name=enc --mem=8G --time=01:10:00 --gres=gpu:1 --array=1-4 \
		code/slurm.sh -- \
    	code/encoding.py -m joint_split --lang-model model-gpt2-2b_layer-24 --cache default_task --save-preds

encoding_bow:
	sbatch --job-name=enc --mem=8G --time=01:10:00 --gres=gpu:1 --array=1-4 \
		code/slurm.sh -- \
    	code/encoding.py -m joint_split --lang-model bow --cache default_task --save-preds

encoding_nollm:
	sbatch --job-name=enc --mem=8G --time=01:10:00 --gres=gpu:1 --array=1-4 \
		code/slurm.sh -- \
    	code/encoding.py -m joint_split_nollm --lang-model model-gpt2-2b_layer-24 --cache default_task --save-preds


encoding_diff_models:
	for space in llm_split; \
		do echo sbatch --job-name=enc --mem=8G --time=03:00:00 --gres=gpu:1 --array=1,2 \
			code/slurm.sh -- \
			code/encoding.py -m "$$space" --lang-model model-gpt2-2b_layer-24 --cache default_task --save-preds; \
	done

encoding_nosplit:
	sbatch --job-name=enc --mem=8G --time=01:10:00 --gres=gpu:1 --array=1-4 \
		code/slurm.sh -- \
    	code/encoding.py -m joint_nosplit --lang-model model-gpt2-2b_layer-24 --cache default_task --save-preds 

encoding_2fold:
	sbatch --job-name=enc2 --mem=16G --time=00:10:00 --gres=gpu:1 --array=1-4 \
		code/slurm.sh -- \
    	code/encoding.py -m joint_split --lang-model model-gpt2-2b_layer-24 --cache default_task --save-preds --suffix _n2

encoding_2fold_weights:
	sbatch --job-name=enc2w --mem=16G --time=01:30:00 --gres=gpu:1 \
		code/slurm.sh -- \
    	code/encoding.py -m joint_nosplit --lang-model model-gpt2-2b_layer-24 --cache default_task --save-preds --suffix _n2w --save-weights

encoding_black2conv:
	sbatch --job-name=b2c --mem=6G --time=00:30:00 --gres=gpu:1 \
		code/slurm.sh -- \
		code/black_encoding.py
