function log(){
  console.log("hello world")
}

const micButton = document.querySelector(".startButton")
const playback = document.getElementsByClassName("playback")

let can_record = false;
let is_recording = false;
let recorder = null;

let audio = null;

let finalWavBlob = null;

let transcript = null;

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
    const blob = new Blob(chunks, {type: "audio/wav"})
    convertBlobToWav(blob);

    const blob_file = new File([blob], "audio.wav");
    chunks = [];

    const audio_url = window.URL.createObjectURL(blob);
    console.log(audio_url)
    playback.src = audio_url;

    audio = new Audio();
    audio.src = audio_url;

    //blobToAPI(blob, "http://127.0.0.1:5000/process_audio");

    console.log(playback.src);
  }

  can_record = true;
}

function convertBlobToWav(blob) {
  // Create a new FileReader
  let reader = new FileReader();

  // Function to handle when the FileReader has loaded the Blob
  reader.onload = function(event) {
      let arrayBuffer = event.target.result;
      let audioContext = new (window.AudioContext || window.webkitAudioContext)();
      
      audioContext.decodeAudioData(arrayBuffer, function(audioBuffer) {
          let wavBuffer = audioBufferToWav(audioBuffer);
          let wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
          finalWavBlob = wavBlob;
      });
  };

  // Read the Blob as an ArrayBuffer
  reader.readAsArrayBuffer(blob);
}

function audioBufferToWav(audioBuffer) {
  let numOfChan = audioBuffer.numberOfChannels,
      length = audioBuffer.length * numOfChan * 2 + 44,
      buffer = new ArrayBuffer(length),
      view = new DataView(buffer),
      channels = [], i, sample,
      offset = 0,
      pos = 0;

  // Write WAV header
  setUint32(0x46464952);                         // "RIFF"
  setUint32(length - 8);                         // file length - 8
  setUint32(0x45564157);                         // "WAVE"

  setUint32(0x20746d66);                         // "fmt " chunk
  setUint32(16);                                 // length = 16
  setUint16(1);                                  // PCM (uncompressed)
  setUint16(numOfChan);
  setUint32(audioBuffer.sampleRate);
  setUint32(audioBuffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
  setUint16(numOfChan * 2);                      // block-align
  setUint16(16);                                 // 16-bit (hardcoded in this demo)

  setUint32(0x61746164);                         // "data" - chunk
  setUint32(length - pos - 4);                   // chunk length

  // Write interleaved data
  for(i = 0; i < audioBuffer.numberOfChannels; i++)
      channels.push(audioBuffer.getChannelData(i));

  while(pos < length) {
      for(i = 0; i < numOfChan; i++) {           // interleave channels
          sample = Math.max(-1, Math.min(1, channels[i][offset])); // clamp
          sample = (sample < 0 ? sample * 0x8000 : sample * 0x7FFF)|0; // scale to 16-bit signed int
          view.setInt16(pos, sample, true);      // write 16-bit sample
          pos += 2;
      }
      offset++;                                  // next source sample
  }

  return buffer;

  function setUint16(data) {
      view.setUint16(pos, data, true);
      pos += 2;
  }

  function setUint32(data) {
      view.setUint32(pos, data, true);
      pos += 4;
  }
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

function uploadWavFile(event) {
  event.preventDefault();
  // Create a new FormData object
  let formData = new FormData();
  localStorage.setItem('src', audio.src);
  // Append the Blob to the FormData object
  formData.append('file', finalWavBlob, 'audio.wav');
  
  // Send the Blob to the server using fetch
  fetch("http://127.0.0.1:5000/process_audio", {
      method: 'POST',
      body: formData
  })
  .then(response => response.json())
  .then(data => {
      console.log('Success:', data);
      console.log(data["message"]);
      transcript = data["message"];
      localStorage.setItem('transcript', transcript);
  })
  .catch((error) => {
      console.error('Error:', error);
  });
}

setupAudio();
if(localStorage.getItem("src") != ""){
  console.log(localStorage.getItem("src"))
  audio = new Audio();
  audio.src = localStorage.getItem("src");
}

let transcriptButton = document.getElementById("pythonScript")
console.log(localStorage.getItem("transcript"));
if(localStorage.getItem("transcript") != ""){
  transcriptButton.innerHTML = localStorage.getItem("transcript");
}

if(audio != null){
  document.getElementById("audioVisualizer").innerHTML = "Audio Loaded";
}