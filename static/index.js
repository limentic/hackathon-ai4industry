// Unique client ID
const clientId = `client_${Date.now()}`;

// Create the WebSocket
const ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`);

// The initial card content
const initialCardBody = `
    <h5 class="card-title text-center">Extractor of image features</h5>
    <form id="uploadForm">
        <div id="dropZone" class="border border-dashed rounded p-3 text-center mb-3" style="cursor: pointer;">
            Drag and drop your image here or click to select
        </div>
        <button type="submit" class="btn btn-primary w-100">Upload</button>
    </form>
`;

const cardBody = document.querySelector(".card-body");
cardBody.innerHTML = initialCardBody;

let selectedFile = null;

// Create a persistent hidden file input for selecting files
const fileInput = document.createElement("input");
fileInput.type = "file";
fileInput.accept = "image/*";
fileInput.style.display = "none";
document.body.appendChild(fileInput);

/**
 * Initialize or re-initialize event listeners on newly replaced elements.
 * This is crucial after we reset the card body.
 */
function initializeEventListeners() {
  // DropZone
  const oldDropZone = document.getElementById("dropZone");
  // Replace the old dropZone element (this ensures old listeners are cleared)
  oldDropZone.replaceWith(oldDropZone.cloneNode(true));

  // Select the new dropZone
  const dropZone = document.getElementById("dropZone");

  // Add dragover event
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  // Add dragleave event
  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  // Add drop event
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      selectedFile = e.dataTransfer.files[0];
      dropZone.textContent = `File Selected: ${selectedFile.name}`;
    }
  });

  // Clicking the dropZone triggers file selection
  dropZone.addEventListener("click", () => {
    fileInput.click();
  });

  // Remove any previous "change" event so we don't stack up multiple listeners
  fileInput.removeEventListener("change", onFileInputChange);
  // Add the "change" event to file input
  fileInput.addEventListener("change", onFileInputChange);

  // Attach the submit event on the (newly re-added) uploadForm
  const uploadForm = document.getElementById("uploadForm");
  uploadForm.addEventListener("submit", onFormSubmit);
}

/**
 * Handler for the file input change event
 */
function onFileInputChange(e) {
  selectedFile = e.target.files[0];
  if (selectedFile) {
    const dropZone = document.getElementById("dropZone");
    dropZone.textContent = `File Selected: ${selectedFile.name}`;
  }
  // Reset the file input value to allow selecting the same file again
  fileInput.value = "";
}

/**
 * Handler for the form submit event
 */
async function onFormSubmit(e) {
  e.preventDefault();
  if (!selectedFile) {
    alert("Please select a file to upload.");
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("client_id", clientId);

  // Update blob colors to dark orange while processing
  updateColors("orange");

  // Show a processing spinner/message
  showProcessing();

  try {
    const response = await fetch("http://localhost:8000/upload/", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();
    console.log("Upload response:", result);
  } catch (err) {
    console.error("Upload error:", err);
    alert("Error uploading file!");
    resetCard();
  }
}

/**
 * WebSocket messages — track the processing status
 */
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.status === "processed") {
    // Processing complete
    updateColors("green");
    displayResult(data.file_id);
  }
};

/**
 * Show a processing spinner in the card
 */
function showProcessing() {
  cardBody.innerHTML = `
    <div class="text-center" style="padding-top: 16px;">
        <div class="spinner-border myorange" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-3">Your media is being processed...</p>
    </div>
  `;
}

/**
 * Fetch and display the result from the server
 */
async function displayResult(fileId) {
  try {
    console.log("Fetching result for file ID:", fileId);
    const response = await fetch(`http://localhost:8000/result/${fileId}`);
    if (!response.ok) throw new Error("Failed to fetch result.");

    const data = await response.json();
    let resultHTML = `
      <table class="table table-bordered">
        <thead>
          <tr>
            <th scope="col">Category</th>
            <th scope="col">Answer</th>
          </tr>
        </thead>
        <tbody>
    `;

    for (const key in data.result) {
      resultHTML += `
        <tr>
          <td><strong>${key}</strong></td>
          <td>${data.result[key]}</td>
        </tr>
      `;
    }

    resultHTML += `
        </tbody>
      </table>
    `;

    cardBody.innerHTML = `
      <h5 id="result-h5" class="card-title text-center">Processing Complete!</h5>
      <div class="text-center">${resultHTML}</div>
      <div class="text-center">
          <button id="resetButton" class="btn btn-secondary mt-3">Reset</button>
      </div>
    `;

    // Reset button to return to initial state
    document.getElementById("resetButton").addEventListener("click", resetCard);
  } catch (error) {
    console.error(error);
    alert("Failed to fetch the result!");
    resetCard();
  }
}

/**
 * Reset card to its initial state
 */
function resetCard() {
  cardBody.innerHTML = initialCardBody;
  updateColors("blue"); // Switch blobs and background to blue
  selectedFile = null;
  initializeEventListeners(); // Rebind fresh listeners on new elements
}

/**
 * Create blob elements and animate them in the background.
 * Also define updateColors to change blob/backdrop colors.
 */
document.addEventListener("DOMContentLoaded", () => {
  const blobContainer = document.getElementById("blobContainer");

  const blueShades = ["#4682B4", "#4169E1", "#1E90FF", "#5F9EA0"]; // Blobs: dark blue
  const greenShades = ["#006400", "#228B22", "#32CD32", "#7FFF00"]; // Blobs: green
  const orangeShades = ["#FF8C00", "#FF4500", "#D2691E", "#A0522D"]; // Blobs: dark orange

  const backgroundColors = {
    blue: "#001f3f", // Background: navy blue
    green: "#013220", // Background: dark green
    orange: "#4B2C20", // Background: deep brownish-orange
  };

  function createBlob() {
    const blob = document.createElement("div");
    blob.classList.add("shape-blob");
    const randomIndex = Math.floor(Math.random() * blueShades.length);
    blob.dataset.colorIndex = randomIndex;
    blob.style.backgroundColor = blueShades[randomIndex]; // start with a blue shade
    blob.style.transition = "background-color 0.4s ease-in-out";
    blob.style.height = `${getRandomSize()}px`;
    blob.style.width = `${getRandomSize()}px`;
    blob.style.top = `${getRandomPosition()}%`;
    blob.style.left = `${getRandomPosition()}%`;
    blobContainer.appendChild(blob);
    animateBlob(blob);
  }

  function getRandomSize() {
    return Math.floor(Math.random() * 180) + 20; // 20px - 200px
  }

  function getRandomPosition() {
    return Math.floor(Math.random() * 100);
  }

  function animateBlob(blob) {
    const duration = Math.random() * 2 + 1; // 1s - 3s
    const keyframes = [
      {
        transform: `translate(${getRandomPosition()}%, ${getRandomPosition()}%)`,
      },
      {
        transform: `translate(${getRandomPosition()}%, ${getRandomPosition()}%)`,
      },
    ];
    blob.animate(keyframes, {
      duration: duration * 1000,
      iterations: Infinity,
      direction: "alternate",
      easing: "ease-in-out",
    });
  }

  function addBlobs() {
    const screenArea = window.innerWidth * window.innerHeight;
    const numBlobs = Math.floor(screenArea / 10000);
    for (let i = 0; i < numBlobs; i++) {
      createBlob();
    }
  }

  // Make updateColors globally accessible
  window.updateColors = function (state) {
    const colorArray =
      state === "green"
        ? greenShades
        : state === "orange"
        ? orangeShades
        : blueShades;

    const backgroundColor = backgroundColors[state];

    // Force a reflow in Safari
    document.body.style.display = "none";
    document.body.offsetHeight;
    document.body.style.display = "";

    // Update each blob's color
    const blobs = document.querySelectorAll(".shape-blob");
    blobs.forEach((blob) => {
      const colorIndex = blob.dataset.colorIndex;
      blob.style.backgroundColor = colorArray[colorIndex];
    });

    // Update the background color
    document.body.style.backgroundColor = backgroundColor;
  };

  // Create and animate blobs
  addBlobs();

  // Set initial colors (blue theme)
  updateColors("blue");

  // Initialize all event listeners the first time
  initializeEventListeners();
});
