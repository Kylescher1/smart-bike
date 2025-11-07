const socket = io("/", {transports:["websocket"], namespace:"/viewer"});
socket.on("dashboard_update", data => {
    if (data.value !== undefined) {
        document.getElementById("numeric").textContent = data.value.toFixed(3);
    }
    if (data.plot) {
        document.getElementById("plot").src = "data:image/png;base64," + data.plot;
    }
    if (data.bike) {
        document.getElementById("raw_data").textContent = JSON.stringify(data.bike, null, 2);
    }
});

// Live plot updater
//function updatePlot() {
//    fetch('/api/plot')
//        .then(res => res.json())
//        .then(data => {
//            document.getElementById('plot').src = "data:image/png;base64," + data.img;
//        });
//}
//setInterval(updatePlot, 2000); // every 2 seconds



// Live data from backend
//socket.on("data_update", (msg) => {
//    console.log("New data:", msg.value);
//});
//socket.on("plot_update", data => {
//    document.getElementById("plot").src = "data:image/png;base64," + data.img;
//});
//
//socket.on("update_dashboard", (data) => {
//    console.log("Live bike data:", data);
//    // Example: dynamically add to an HTML element
//    document.getElementById("live-data").innerText = JSON.stringify(data, null, 2);
//});
