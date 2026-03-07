import whisper
import os
import json
model = whisper.load_model("large")

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
audio_dir = os.path.join(parent_dir, "audios")
audios = os.listdir(audio_dir)

result = model.transcribe(f"{audio_dir}/11-Video_11.mp3")
chunks = []
for segment in result["segments"]:
    chunks.append(
        {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
    )
# print(result["text"]);
# print(result);
# print(chunks);
output_path=os.path.join(curr_dir, "11-Video_11.json")
with open(output_path, "w") as f:
    json.dump(chunks, f, indent=4);

