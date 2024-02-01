export {restartVideo, shutDownVideo}

function restartVideo() {
    var videoElement = document.getElementById('videoStream');
    // Append a unique query parameter to the src to force reload
    var uniqueSrc = "{{ url_for('home_blueprint.video') }}" + "?t=" + new Date().getTime();
    videoElement.src = uniqueSrc; // Reset the src to restart the video
    videoElement.style.display = '';
}

function shutDownVideo() {
    // Send a request to the Flask server to shut down the video stream
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "{{ url_for('home_blueprint.shutdown_video') }}", true);
    xhr.send();

    // Optionally, hide the video element or display a placeholder
    if (document.getElementById('videoStream').style.display == 'none') {
        console.log("resart")
        restartVideo()
    } else {
        document.getElementById('videoStream').style.display = 'none';
        console.log("close")
    }
}