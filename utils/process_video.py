# converts the videos to mp3

import subprocess;
import os;

# files = os.listdir('../videos');
# print(files);

# Get the directory where this script is located
# os.path.abspath(__file__) -- returns absolute path of the file
#os.path.dirname(os.path.abspath(__file__)) returns the current directory 
current_dir = os.path.dirname(os.path.abspath(__file__))

#Going one level up
parent_dir=os.path.dirname(current_dir);

videos_dir=os.path.join(parent_dir, "videos");

files=os.listdir(videos_dir);

for file in files:
    if file.endswith('.mp4'):
        # print(os.path.splitext(file));
        name_without_ext=os.path.splitext(file)[0];
        
        parts=name_without_ext.split('-');
        # print(parts);
        
        audio_number=parts[0];
        audio_title=parts[1];
        # print(f"{audio_number} & {audio_title}")

        #input and output paths
        input_path=os.path.join(videos_dir, file);
        audios_dir=os.path.join(parent_dir, "audios");
        output_file_name=f"{audio_number}-{audio_title}.mp3";
        output_path=os.path.join(audios_dir, f"{output_file_name}")

        #using subprocess to convert vidoes files to audio files
        subprocess.run([
            "ffmpeg",
            "-i",
            f"{input_path}",
            f"{output_path}"
        ]);

        # audio_number=file.split('-')[0];
        # audio_title=file.split('-')[1];
        # # print(f"The current audio number is {audio_number}");
        # # print(f"The current audio number and tit""le are {audio_number} and {audio_title}");
        # with open(file, 'w') as f:
        #     subprocess.run(["ffmpeg", "-i", f"{parent_dir}/videos", f"{parent_dir}/audios/{}"],)

# with open('../videos/01-Video-1.mp4', 'r') as f:
    # subprocess.run('ffmg')