alias spawn_ezsdr="docker run -i --rm --init --net=host $(lsusb | grep "HackRF" | awk '{printf "--device=/dev/bus/usb/%03d/%03d ", $2, $4}') ghcr.io/k3kaimu/ezsdr:v3.0.14"
