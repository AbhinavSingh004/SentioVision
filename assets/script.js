// assets/script.js
function updateClock() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-IN', { hour12: true });
    const clockElement = document.getElementById('live-clock');
    if (clockElement) {
        clockElement.innerText = "System Time: " + timeString;
    }
}
setInterval(updateClock, 1000);
updateClock();