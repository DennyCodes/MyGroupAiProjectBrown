function log(){
  console.log("hello world")
}

const micButton = document.querySelector(".startButton")
const playback = document.getElementsByClassName("playback")

let can_record = false;
let is_recording = false;
let recorder = null;

let audio = null;


let chunks = [];

function setupAudio(){
  console.log("setup")
  if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia){
    navigator.mediaDevices
      .getUserMedia({
        audio: true
      })
      .then(setupAudioStream)
      .catch(err => {console.error(err)} )
  }
  else{
    alert("No user media available, try again")
  }
}


function toggleMic(){
  if(can_record){
    is_recording = !is_recording;

    if(is_recording){
      recorder.start();
      console.log("started recording")
    }
    else{
      recorder.stop();
      console.log("stopped recording")
    }
  }
  else{
    return
  }
}

function setupAudioStream(stream){
  recorder = new MediaRecorder(stream)

  recorder.ondataavailable = e => {
    chunks.push(e.data);
  }

  recorder.onstop = async e => {
    const blob = new Blob(chunks, {type: "audio/mp3"})
    chunks = [];

    const audio_url = window.URL.createObjectURL(blob);
    console.log(audio_url)
    playback.src = audio_url;

    audio = new Audio();
    audio.src = audio_url;

    //blobToAPI(blob, "http://127.0.0.1:5000/process_audio");
    const res = await uploadBlob(audio);
    console.log(playback.src);
  }

  can_record = true;
}

function play_audio(){
  if(audio != null){
    audio.play().catch(function(err){
      console.error(err)
    })
    audio.addEventListener("ended", () => {
      btn.classList.add("fa-play")
      btn.classList.remove("fa-pause")
    })

    let btn = document.getElementById("play_pause");
    btn.classList.remove("fa-play")
    btn.classList.add("fa-pause")
  }
}

async function uploadBlob(audioBlob, fileType) {
  const formData = new FormData();
  formData.append('audio_data', audioBlob, 'file');
  formData.append('type', fileType || 'mp3');

  // Your server endpoint to upload audio:
  const apiUrl = "http://127.0.0.1:5000/process_audio";

  const response = await fetch(apiUrl, {
    method: 'POST',
    cache: 'no-cache',
    body: formData
  });

  return response.json();
}


setupAudio();