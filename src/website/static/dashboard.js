const socket = io();

// Live plot updater
function updatePlot() {
    fetch('/api/plot')
        .then(res => res.json())
        .then(data => {
            document.getElementById('plot').src = "data:image/png;base64," + data.img;
        });
}
setInterval(updatePlot, 2000); // every 2 seconds

// Live data from backend
socket.on("data_update", (msg) => {
    console.log("New data:", msg.value);
});
