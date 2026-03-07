import whisper
import os
import json
model = whisper.load_model("large")

# def create_chunk_each_audio(audio, audio_dir):


curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
audio_dir = os.path.join(parent_dir, "audios")
audios = os.listdir(audio_dir)
json_dir=os.path.join(parent_dir, "json_files")

# audio_file='01-Video_1.mp3'
# j_sample=os.path.splitext(audio_file)[0]+'.json';
# print(j_sample)

for audio in audios:
    audio_without_ext=os.path.splitext(audio)[0];
    json_filename=audio_without_ext+'.json';
    number=audio_without_ext.split('-')[0]; 
    title=audio_without_ext.split('-')[1]; 
    result = model.transcribe(f"{audio_dir}/{audio}")
    chunks = []
    for segment in result["segments"]:
        chunks.append(
            {"number":number, "title":title ,"start": segment["start"], "end": segment["end"], "text": segment["text"]}
        )
    chunks_with_metadata={"chunks":chunks, "text":result["text"]}
    with open(f"{json_dir}/{json_filename}", "w") as f:
        json.dump(chunks_with_metadata, f);    


   


