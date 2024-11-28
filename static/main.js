


// ===== LAYERS INPUTS =====

const layersInputsContainer = document.getElementById("layers-inputs")

// layersCounter is the number of layers the user has added
let layersCounter = 0

function createNodeInputs(event, deltaLayers) {

    // Prevents error on initial run on DOMContentLoaded due to no event
    if (event) {
        event.preventDefault()
    }

    // 1 <= layersCounter <= 8 results in a max of 10 layers (incl. input & output) to improve performance and ensure there is always a layer between the input and outut layers 
    layersCounter = Math.min(Math.max(layersCounter + deltaLayers, 1), 8)

    // Create input nodes field
    layersInputsContainer.innerHTML = `
        <label for="layer-0-nodes" class="form-label">Layer 0 (input) nodes:</label>
        <input type="number" id="layer-0-nodes" name="layer0Nodes" class="form-control" value="784" readonly>
    `
    
    // Create layer nodes fields for layersCounter layers
    for (let i=1; i<=layersCounter; i++) {
        layersInputsContainer.innerHTML += `
            <label for="layer-${i}-nodes" class="form-label">Layer ${i} nodes:</label>
            <input type="number" id="layer-${i}-nodes" name="layer${i}Nodes" class="form-control" required>
        `
    }

    // Create output nodes field
    layersInputsContainer.innerHTML += `
        <label for="layer-${layersCounter+1}-nodes" class="form-label">Layer ${layersCounter+1} (output) nodes:</label>
        <input type="number" id="layer-${layersCounter+1}-nodes" name="layer${layersCounter+1}Nodes" class="form-control" value="10" readonly>
    `

}

document.addEventListener("DOMContentLoaded", createNodeInputs(null, 1))



// ===== IMAGE LOADING =====

const results = document.getElementById("results")
const pixelsContainer = document.getElementById("image")
const currentTestingDataAccuracy = document.getElementById("current-testing-data-accuracy")
const currentImageIndex = document.getElementById("current-image-index")
const currentImageActual = document.getElementById("current-image-actual")
const currentImagePrediction = document.getElementById("current-image-prediction")
let imageData
let index = 0

// Returns JSON with form { "index" : {"actual" : actual, "index" : index, "pixels" : [pixels]}, ... }
async function fetchImageData() {
    let response = await fetch("/getImageData")
    let imageData = await response.json()
    return imageData
}

function drawImage(pixels) {
    pixelsContainer.innerHTML = ""

    // Add pixels to fragment and add fragment to document to improve performance
    let fragment = document.createDocumentFragment()
    for (let i=0; i<pixels.length; i++) {
        const pixel = document.createElement("div")
        pixel.style.backgroundColor = `rgb(${pixels[i]*255}, ${pixels[i]*255}, ${pixels[i]*255})`
        fragment.appendChild(pixel)
    }
    pixelsContainer.appendChild(fragment)
}

function updateImage(imageData, newImageIndex) {
    currentImageData = imageData[`${newImageIndex}`]

    index = currentImageData["index"]
    actual = currentImageData["actual"]
    pixels = currentImageData["pixels"]

    currentImageIndex.innerHTML = index
    currentImageActual.innerHTML = actual

    drawImage(pixels)
}

function changeImage(event, deltaIndex) {
    event.preventDefault()

    // 0 <= index <= 999 because this is the range of available indices for viewing
    index = Math.min(Math.max(index + deltaIndex, 0), 999)

    updateImage(imageData, index)
}

document.addEventListener("DOMContentLoaded", async () => {
    imageData = await fetchImageData()
    updateImage(imageData, index)
})



// ===== AESTHETIC =====

function changeTheme(theme) {
    results.classList.forEach(className => {
        if (className.startsWith("border-")) {
            results.classList.remove(className)
        }
    })
    results.classList.add(`border-${theme}`)
}



// ===== RECORDS =====

const recordsContainer = document.getElementById("records")

function createRecord(alpha, i, layers, finalAccuracy, peakAccuracy) {
    recordsContainer.innerHTML += `
        <div class="bg-body-tertiary border border-light-subtle p-4 rounded-4 mb-4">
            <div>Learning rate: ${alpha}</div>
            <div>Iterations: ${i}</div>
            <div>Layer structure: ${layers}</div>
            <div>Final testing data accuracy: ${finalAccuracy}%</div>
            <div>Peak testing data accuracy: ${peakAccuracy}%</div>
        </div>
    `
}