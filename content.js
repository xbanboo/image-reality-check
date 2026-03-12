// Content script: listens for context menu trigger, opens analyzer panel
// v2.0 - Added TensorFlow.js face detection + AI-generated image detection

let tfLoaded = false;
let tfLoadFailed = false;
let blazefaceModel = null;

// Suppress CSP EvalError from TF.js (non-fatal, falls back to heuristic detection)
window.addEventListener('error', (e) => {
  if (e.error instanceof EvalError && e.message && e.message.includes('Content Security Policy')) {
    e.preventDefault();
    tfLoadFailed = true;
  }
});

window.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'REALITY_CHECK_OPEN') {
    openAnalyzerPanel(event.data.imageUrl);
  }
});

// ===== TensorFlow.js Loading =====
async function loadTF() {
  if (tfLoaded) return true;
  if (tfLoadFailed) return false;
  try {
    // Test if page CSP allows script injection
    const testScript = document.createElement('script');
    testScript.textContent = 'void(0)';
    document.head.appendChild(testScript);
    testScript.remove();
  } catch (e) {
    console.warn('CSP blocks inline scripts, skipping TF.js');
    tfLoadFailed = true;
    return false;
  }
  try {
    const tfScript = document.createElement('script');
    tfScript.src = chrome.runtime.getURL('lib/tf.min.js');
    document.head.appendChild(tfScript);
    await new Promise((resolve, reject) => {
      tfScript.onload = resolve;
      tfScript.onerror = reject;
    });
    // Verify TF.js actually loaded (CSP may block eval internally)
    if (!window.tf) { tfLoadFailed = true; return false; }
    tfLoaded = true;
    return true;
  } catch (e) {
    console.warn('TF.js load failed:', e);
    tfLoadFailed = true;
    return false;
  }
}

async function loadBlazeFace() {
  if (blazefaceModel) return blazefaceModel;
  try {
    await loadTF();
    if (!window.tf) return null;

    const bfScript = document.createElement('script');
    bfScript.src = chrome.runtime.getURL('lib/blazeface.min.js');
    document.head.appendChild(bfScript);
    await new Promise((resolve, reject) => {
      bfScript.onload = resolve;
      bfScript.onerror = reject;
    });

    // Use the blazeface library's proper load() API with local model
    const modelUrl = chrome.runtime.getURL('lib/blazeface-model/model.json');
    blazefaceModel = await window.blazeface.load({ modelUrl });
    return blazefaceModel;
  } catch (e) {
    console.warn('BlazeFace load failed:', e);
    return null;
  }
}

// ===== BlazeFace Face Detection =====
async function detectFacesBlazeFace(canvas) {
  const model = await loadBlazeFace();
  if (!model) return [];

  try {
    // Use the proper estimateFaces API
    const predictions = await model.estimateFaces(canvas, false);

    const faces = [];
    for (const pred of predictions) {
      const topLeft = pred.topLeft;
      const bottomRight = pred.bottomRight;
      const x = topLeft[0];
      const y = topLeft[1];
      const w = bottomRight[0] - topLeft[0];
      const h = bottomRight[1] - topLeft[1];

      // Filter: face must be reasonable size (at least 1% of image)
      const faceArea = w * h;
      const imageArea = canvas.width * canvas.height;
      if (faceArea < imageArea * 0.001) continue;

      faces.push({
        x, y, width: w, height: h,
        score: pred.probability[0] || pred.probability,
        landmarks: pred.landmarks || []
      });
    }
    return faces;
  } catch (e) {
    console.warn('BlazeFace detection failed:', e);
    return [];
  }
}

// ===== Fallback: Heuristic Face Detection =====
function detectFacesHeuristic(canvas, ctx) {
  const w = canvas.width, h = canvas.height;
  const data = ctx.getImageData(0, 0, w, h).data;
  const skinMask = new Uint8Array(w * h);

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    if (r > 95 && g > 40 && b > 20 && r > g && r > b && (r - g) > 15 && r - Math.min(g, b) > 15) {
      skinMask[i / 4] = 1;
    }
  }

  // Find connected skin regions using simple scanning
  const faces = [];
  const blockSize = Math.max(8, Math.floor(Math.min(w, h) / 40));
  const blocksX = Math.ceil(w / blockSize);
  const blocksY = Math.ceil(h / blockSize);
  const blockSkin = new Float32Array(blocksX * blocksY);

  for (let by = 0; by < blocksY; by++) {
    for (let bx = 0; bx < blocksX; bx++) {
      let count = 0, total = 0;
      for (let y = by * blockSize; y < Math.min((by + 1) * blockSize, h); y++) {
        for (let x = bx * blockSize; x < Math.min((bx + 1) * blockSize, w); x++) {
          total++;
          if (skinMask[y * w + x]) count++;
        }
      }
      blockSkin[by * blocksX + bx] = count / total;
    }
  }

  // Find clusters of high-skin blocks (>40% skin)
  const visited = new Uint8Array(blocksX * blocksY);
  for (let by = 0; by < blocksY; by++) {
    for (let bx = 0; bx < blocksX; bx++) {
      const idx = by * blocksX + bx;
      if (visited[idx] || blockSkin[idx] < 0.4) continue;

      let minX = bx, maxX = bx, minY = by, maxY = by;
      const queue = [idx];
      visited[idx] = 1;
      let clusterSize = 0;

      while (queue.length > 0) {
        const ci = queue.shift();
        const cy = Math.floor(ci / blocksX);
        const cx = ci % blocksX;
        clusterSize++;
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        for (const [dx, dy] of [[0, 1], [0, -1], [1, 0], [-1, 0]]) {
          const nx = cx + dx, ny = cy + dy;
          if (nx < 0 || nx >= blocksX || ny < 0 || ny >= blocksY) continue;
          const ni = ny * blocksX + nx;
          if (!visited[ni] && blockSkin[ni] >= 0.3) {
            visited[ni] = 1;
            queue.push(ni);
          }
        }
      }

      const fw = (maxX - minX + 1) * blockSize;
      const fh = (maxY - minY + 1) * blockSize;
      const aspectRatio = fh / fw;

      // Tighter constraints: require larger clusters, more face-like aspect ratio, minimum size
      if (clusterSize >= 8 && aspectRatio > 1.0 && aspectRatio < 1.8 && fw > w * 0.06 && fh > h * 0.08) {
        faces.push({
          x: minX * blockSize,
          y: minY * blockSize,
          width: fw,
          height: fh,
          score: Math.min(1, clusterSize / 20),
          landmarks: []
        });
      }
    }
  }

  return faces;
}

// ===== AI-Generated Image Detection =====

function analyzeFrequencyDomain(canvas, ctx) {
  const size = 256;
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = size;
  tempCanvas.height = size;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(canvas, 0, 0, size, size);
  const data = tempCtx.getImageData(0, 0, size, size).data;

  // Convert to grayscale
  const gray = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) {
    gray[i] = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3;
  }

  // Simple 2D DFT on downsampled version for performance
  const dftSize = 64;
  const small = new Float32Array(dftSize * dftSize);
  const step = size / dftSize;
  for (let y = 0; y < dftSize; y++) {
    for (let x = 0; x < dftSize; x++) {
      small[y * dftSize + x] = gray[Math.floor(y * step) * size + Math.floor(x * step)];
    }
  }

  // Compute magnitude spectrum using row/col DFT
  const spectrum = new Float32Array(dftSize * dftSize);
  const PI2 = 2 * Math.PI;

  // Row-wise DFT
  const rowReal = new Float32Array(dftSize * dftSize);
  const rowImag = new Float32Array(dftSize * dftSize);
  for (let y = 0; y < dftSize; y++) {
    for (let k = 0; k < dftSize; k++) {
      let re = 0, im = 0;
      for (let x = 0; x < dftSize; x++) {
        const angle = PI2 * k * x / dftSize;
        re += small[y * dftSize + x] * Math.cos(angle);
        im -= small[y * dftSize + x] * Math.sin(angle);
      }
      rowReal[y * dftSize + k] = re;
      rowImag[y * dftSize + k] = im;
    }
  }

  // Column-wise DFT
  for (let k2 = 0; k2 < dftSize; k2++) {
    for (let k1 = 0; k1 < dftSize; k1++) {
      let re = 0, im = 0;
      for (let y = 0; y < dftSize; y++) {
        const angle = PI2 * k2 * y / dftSize;
        const rr = rowReal[y * dftSize + k1];
        const ri = rowImag[y * dftSize + k1];
        re += rr * Math.cos(angle) - ri * Math.sin(angle);
        im += rr * (-Math.sin(angle)) + ri * Math.cos(angle);
      }
      spectrum[k2 * dftSize + k1] = Math.sqrt(re * re + im * im);
    }
  }

  // Analyze frequency distribution - compute radial power spectrum
  const half = dftSize / 2;
  const radialBins = half;
  const radialPower = new Float32Array(radialBins);
  const radialCount = new Float32Array(radialBins);

  for (let y = 0; y < dftSize; y++) {
    for (let x = 0; x < dftSize; x++) {
      const dy = y < half ? y : y - dftSize;
      const dx = x < half ? x : x - dftSize;
      const r = Math.sqrt(dx * dx + dy * dy);
      const bin = Math.min(Math.floor(r), radialBins - 1);
      radialPower[bin] += spectrum[y * dftSize + x];
      radialCount[bin]++;
    }
  }

  for (let i = 0; i < radialBins; i++) {
    if (radialCount[i] > 0) radialPower[i] /= radialCount[i];
  }

  // AI images tend to have steeper falloff in high frequencies
  // Natural images follow ~1/f power law more closely
  const lowFreqPower = radialPower.slice(1, 8).reduce((a, b) => a + b, 0);
  const midFreqPower = radialPower.slice(8, 20).reduce((a, b) => a + b, 0);
  const highFreqPower = radialPower.slice(20).reduce((a, b) => a + b, 0);

  const totalPower = lowFreqPower + midFreqPower + highFreqPower;
  const highFreqRatio = highFreqPower / (totalPower || 1);
  const midFreqRatio = midFreqPower / (totalPower || 1);

  // AI images: lower high-freq ratio, more concentrated in low frequencies
  // Real photos: more distributed across frequencies
  // Tightened thresholds - be more skeptical
  const freqScore = highFreqRatio < 0.03 ? 90 :
    highFreqRatio < 0.08 ? 75 :
    highFreqRatio < 0.15 ? 55 :
    highFreqRatio < 0.25 ? 35 : 15;

  // Generate spectrum visualization canvas (color)
  const vizCanvas = document.createElement('canvas');
  vizCanvas.width = dftSize;
  vizCanvas.height = dftSize;
  const vizCtx = vizCanvas.getContext('2d');
  const vizData = vizCtx.createImageData(dftSize, dftSize);

  const maxSpec = Math.max(...spectrum) || 1;
  for (let i = 0; i < dftSize * dftSize; i++) {
    // Shift zero frequency to center
    const y = i / dftSize | 0;
    const x = i % dftSize;
    const sy = (y + half) % dftSize;
    const sx = (x + half) % dftSize;
    const val = Math.log(1 + spectrum[sy * dftSize + sx]) / Math.log(1 + maxSpec);
    // Color spectrum: blue→cyan→green→yellow→red
    const t = val;
    let sr, sg, sb;
    if (t < 0.25) { sr = 0; sg = Math.round(t * 4 * 255); sb = 255; }
    else if (t < 0.5) { sr = 0; sg = 255; sb = Math.round((1 - (t - 0.25) * 4) * 255); }
    else if (t < 0.75) { sr = Math.round((t - 0.5) * 4 * 255); sg = 255; sb = 0; }
    else { sr = 255; sg = Math.round((1 - (t - 0.75) * 4) * 255); sb = 0; }
    vizData.data[i * 4] = sr;
    vizData.data[i * 4 + 1] = sg;
    vizData.data[i * 4 + 2] = sb;
    vizData.data[i * 4 + 3] = 255;
  }
  vizCtx.putImageData(vizData, 0, 0);

  return {
    freqScore,
    highFreqRatio,
    midFreqRatio,
    radialPower,
    spectrumCanvas: vizCanvas
  };
}

function analyzeNoisePattern(canvas, ctx) {
  const w = canvas.width, h = canvas.height;
  const data = ctx.getImageData(0, 0, w, h).data;

  // Extract noise by high-pass filtering (subtract blurred version)
  const blockSize = 16;
  const blocksX = Math.ceil(w / blockSize);
  const blocksY = Math.ceil(h / blockSize);
  const blockNoiseStd = [];

  for (let by = 0; by < blocksY; by++) {
    for (let bx = 0; bx < blocksX; bx++) {
      const noiseVals = [];
      for (let y = by * blockSize + 1; y < Math.min((by + 1) * blockSize, h) - 1; y++) {
        for (let x = bx * blockSize + 1; x < Math.min((bx + 1) * blockSize, w) - 1; x++) {
          const idx = (y * w + x) * 4;
          // High-pass: pixel - average of neighbors
          for (let c = 0; c < 3; c++) {
            const center = data[idx + c];
            const avg = (data[idx - 4 + c] + data[idx + 4 + c] +
              data[((y - 1) * w + x) * 4 + c] + data[((y + 1) * w + x) * 4 + c]) / 4;
            noiseVals.push(center - avg);
          }
        }
      }

      if (noiseVals.length > 0) {
        const mean = noiseVals.reduce((a, b) => a + b, 0) / noiseVals.length;
        const std = Math.sqrt(noiseVals.reduce((a, b) => a + (b - mean) ** 2, 0) / noiseVals.length);
        blockNoiseStd.push(std);
      }
    }
  }

  // AI images tend to have very uniform noise across blocks
  const meanStd = blockNoiseStd.reduce((a, b) => a + b, 0) / blockNoiseStd.length;
  const stdOfStd = Math.sqrt(blockNoiseStd.reduce((a, b) => a + (b - meanStd) ** 2, 0) / blockNoiseStd.length);
  const coeffOfVariation = stdOfStd / (meanStd || 1);

  // Low variation = uniform noise = suspicious (AI-generated)
  // High variation = natural noise patterns
  // Tightened: most AI images fall in 0.1-0.3 range
  const noiseScore = coeffOfVariation < 0.12 ? 90 :
    coeffOfVariation < 0.2 ? 70 :
    coeffOfVariation < 0.35 ? 50 :
    coeffOfVariation < 0.5 ? 30 : 10;

  return { noiseScore, coeffOfVariation, meanNoiseStd: meanStd };
}

function analyzeColorHistogram(canvas, ctx) {
  const w = canvas.width, h = canvas.height;
  const data = ctx.getImageData(0, 0, w, h).data;

  const histR = new Float32Array(256);
  const histG = new Float32Array(256);
  const histB = new Float32Array(256);
  const pixels = w * h;

  for (let i = 0; i < data.length; i += 4) {
    histR[data[i]]++;
    histG[data[i + 1]]++;
    histB[data[i + 2]]++;
  }

  // Normalize
  for (let i = 0; i < 256; i++) {
    histR[i] /= pixels;
    histG[i] /= pixels;
    histB[i] /= pixels;
  }

  // Calculate smoothness of histogram (AI images have smoother distributions)
  function histSmoothness(hist) {
    let totalDiff = 0;
    for (let i = 1; i < 255; i++) {
      totalDiff += Math.abs(hist[i + 1] - 2 * hist[i] + hist[i - 1]);
    }
    return totalDiff;
  }

  const smoothR = histSmoothness(histR);
  const smoothG = histSmoothness(histG);
  const smoothB = histSmoothness(histB);
  const avgSmooth = (smoothR + smoothG + smoothB) / 3;

  // Very smooth histogram = AI, jagged = natural
  // Tightened thresholds
  const colorScore = avgSmooth < 0.0003 ? 85 :
    avgSmooth < 0.0008 ? 70 :
    avgSmooth < 0.002 ? 50 :
    avgSmooth < 0.004 ? 30 : 10;

  // Check for unusual color saturation uniformity
  let satValues = [];
  for (let i = 0; i < data.length; i += 16) { // sample every 4th pixel
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    if (max > 0) satValues.push((max - min) / max);
  }
  const meanSat = satValues.reduce((a, b) => a + b, 0) / satValues.length;
  const satStd = Math.sqrt(satValues.reduce((a, b) => a + (b - meanSat) ** 2, 0) / satValues.length);

  return { colorScore, avgSmooth, satMean: meanSat, satStd };
}

function analyzeTextureRegularity(canvas, ctx) {
  const w = canvas.width, h = canvas.height;
  const data = ctx.getImageData(0, 0, w, h).data;

  // Compute LBP (Local Binary Pattern) histogram for texture analysis
  const lbpHist = new Float32Array(256);
  let count = 0;

  const step = Math.max(1, Math.floor(Math.min(w, h) / 256)); // downsample for speed

  for (let y = 1; y < h - 1; y += step) {
    for (let x = 1; x < w - 1; x += step) {
      const idx = y * w + x;
      const center = (data[idx * 4] + data[idx * 4 + 1] + data[idx * 4 + 2]) / 3;

      let lbp = 0;
      const neighbors = [
        (y - 1) * w + (x - 1), (y - 1) * w + x, (y - 1) * w + (x + 1),
        y * w + (x + 1), (y + 1) * w + (x + 1), (y + 1) * w + x,
        (y + 1) * w + (x - 1), y * w + (x - 1)
      ];

      for (let n = 0; n < 8; n++) {
        const ni = neighbors[n];
        const nVal = (data[ni * 4] + data[ni * 4 + 1] + data[ni * 4 + 2]) / 3;
        if (nVal >= center) lbp |= (1 << n);
      }

      lbpHist[lbp]++;
      count++;
    }
  }

  // Normalize
  for (let i = 0; i < 256; i++) lbpHist[i] /= count;

  // Calculate entropy of LBP histogram
  let entropy = 0;
  for (let i = 0; i < 256; i++) {
    if (lbpHist[i] > 0) entropy -= lbpHist[i] * Math.log2(lbpHist[i]);
  }

  // AI images often have lower LBP entropy (more regular textures)
  // Natural images: higher entropy (more varied textures)
  // Tightened: real photos typically have entropy > 6.5
  const textureScore = entropy < 4.5 ? 85 :
    entropy < 5.5 ? 70 :
    entropy < 6.2 ? 50 :
    entropy < 7 ? 30 : 12;

  return { textureScore, lbpEntropy: entropy };
}

function analyzeSymmetry(canvas, ctx, faces) {
  if (!faces || faces.length === 0) return { symmetryScore: 30, symmetryValue: 0.5 };

  const w = canvas.width, h = canvas.height;
  const data = ctx.getImageData(0, 0, w, h).data;

  let totalSymmetry = 0;
  let faceCount = 0;

  for (const face of faces) {
    const fx = Math.max(0, Math.floor(face.x));
    const fy = Math.max(0, Math.floor(face.y));
    const fw = Math.min(Math.floor(face.width), w - fx);
    const fh = Math.min(Math.floor(face.height), h - fy);
    if (fw < 10 || fh < 10) continue;

    const halfW = Math.floor(fw / 2);
    let diffSum = 0, count = 0;

    for (let y = fy; y < fy + fh; y++) {
      for (let x = 0; x < halfW; x++) {
        const leftIdx = (y * w + (fx + x)) * 4;
        const rightIdx = (y * w + (fx + fw - 1 - x)) * 4;
        if (leftIdx >= 0 && rightIdx >= 0 && leftIdx < data.length && rightIdx < data.length) {
          for (let c = 0; c < 3; c++) {
            diffSum += Math.abs(data[leftIdx + c] - data[rightIdx + c]);
          }
          count++;
        }
      }
    }

    const avgDiff = count > 0 ? diffSum / (count * 3) : 50;
    totalSymmetry += avgDiff;
    faceCount++;
  }

  if (faceCount === 0) return { symmetryScore: 30, symmetryValue: 0.5 };

  const avgSymmetry = totalSymmetry / faceCount;
  // Very symmetrical faces (low diff) = more likely AI
  // Asymmetrical (high diff) = more likely real
  // Tightened: real faces typically have asymmetry > 12
  const symmetryScore = avgSymmetry < 6 ? 85 :
    avgSymmetry < 10 ? 70 :
    avgSymmetry < 16 ? 50 :
    avgSymmetry < 25 ? 30 : 12;

  return { symmetryScore, symmetryValue: avgSymmetry };
}

// ===== Screenshot Detection =====
function detectScreenshot(canvas, ctx) {
  const w = canvas.width, h = canvas.height;
  const data = ctx.getImageData(0, 0, w, h).data;
  const totalPixels = w * h;

  // 1. Flat color block ratio: sample blocks, check if pixels are near-identical
  const blockSize = 16;
  const blocksX = Math.ceil(w / blockSize);
  const blocksY = Math.ceil(h / blockSize);
  let flatBlocks = 0, totalBlocks = 0;

  for (let by = 0; by < blocksY; by++) {
    for (let bx = 0; bx < blocksX; bx++) {
      let minR = 255, maxR = 0, minG = 255, maxG = 0, minB = 255, maxB = 0;
      for (let y = by * blockSize; y < Math.min((by + 1) * blockSize, h); y++) {
        for (let x = bx * blockSize; x < Math.min((bx + 1) * blockSize, w); x++) {
          const i = (y * w + x) * 4;
          const r = data[i], g = data[i + 1], b = data[i + 2];
          if (r < minR) minR = r; if (r > maxR) maxR = r;
          if (g < minG) minG = g; if (g > maxG) maxG = g;
          if (b < minB) minB = b; if (b > maxB) maxB = b;
        }
      }
      totalBlocks++;
      // A flat block has very low color range (< 5 per channel)
      if ((maxR - minR) < 5 && (maxG - minG) < 5 && (maxB - minB) < 5) flatBlocks++;
    }
  }
  const flatRatio = flatBlocks / totalBlocks;

  // 2. Unique color count in sample: screenshots use limited palette
  const colorSet = new Set();
  const sampleStep = Math.max(1, Math.floor(totalPixels / 10000));
  for (let i = 0; i < data.length; i += sampleStep * 4) {
    // Quantize to reduce near-duplicates
    const r = data[i] >> 2, g = data[i + 1] >> 2, b = data[i + 2] >> 2;
    colorSet.add((r << 16) | (g << 8) | b);
  }
  const uniqueColors = colorSet.size;
  const lowColorDiversity = uniqueColors < 500;

  // 3. Noise level: screenshots have near-zero high-frequency noise
  let noiseSum = 0, noiseCount = 0;
  for (let y = 1; y < h - 1; y += 3) {
    for (let x = 1; x < w - 1; x += 3) {
      const idx = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) {
        const center = data[idx + c];
        const avg = (data[idx - 4 + c] + data[idx + 4 + c] +
          data[((y - 1) * w + x) * 4 + c] + data[((y + 1) * w + x) * 4 + c]) / 4;
        noiseSum += Math.abs(center - avg);
        noiseCount++;
      }
    }
  }
  const avgNoise = noiseSum / noiseCount;
  const veryLowNoise = avgNoise < 1.5;

  // Decision: screenshot if 2+ indicators positive
  let indicators = 0;
  if (flatRatio > 0.35) indicators++;
  if (lowColorDiversity) indicators++;
  if (veryLowNoise) indicators++;
  if (flatRatio > 0.5) indicators++; // strong signal

  const isScreenshot = indicators >= 2;

  return { isScreenshot, flatRatio, uniqueColors, avgNoise, indicators };
}

function analyzeEdgeCoherence(canvas, ctx) {
  const w = canvas.width, h = canvas.height;
  const data = ctx.getImageData(0, 0, w, h).data;

  // Compute edge map using Sobel
  const edgeStrengths = [];
  const edgeDirections = [];

  for (let y = 1; y < h - 1; y += 2) {
    for (let x = 1; x < w - 1; x += 2) {
      const getG = (px, py) => {
        const i = (py * w + px) * 4;
        return (data[i] + data[i + 1] + data[i + 2]) / 3;
      };

      const gx = -getG(x - 1, y - 1) - 2 * getG(x - 1, y) - getG(x - 1, y + 1)
        + getG(x + 1, y - 1) + 2 * getG(x + 1, y) + getG(x + 1, y + 1);
      const gy = -getG(x - 1, y - 1) - 2 * getG(x, y - 1) - getG(x + 1, y - 1)
        + getG(x - 1, y + 1) + 2 * getG(x, y + 1) + getG(x + 1, y + 1);

      const mag = Math.sqrt(gx * gx + gy * gy);
      if (mag > 20) {
        edgeStrengths.push(mag);
        edgeDirections.push(Math.atan2(gy, gx));
      }
    }
  }

  if (edgeStrengths.length < 10) return { edgeScore: 30, edgeCoherence: 0.5 };

  // Analyze edge direction histogram - AI images may have more uniform edge distributions
  const dirBins = 36;
  const dirHist = new Float32Array(dirBins);
  for (const dir of edgeDirections) {
    const bin = Math.floor(((dir + Math.PI) / (2 * Math.PI)) * dirBins) % dirBins;
    dirHist[bin]++;
  }

  // Normalize
  const totalEdges = edgeStrengths.length;
  for (let i = 0; i < dirBins; i++) dirHist[i] /= totalEdges;

  // Calculate entropy of edge directions
  let dirEntropy = 0;
  for (let i = 0; i < dirBins; i++) {
    if (dirHist[i] > 0) dirEntropy -= dirHist[i] * Math.log2(dirHist[i]);
  }

  // AI images: edges tend to be less sharp / more blurred at boundaries
  const meanEdge = edgeStrengths.reduce((a, b) => a + b, 0) / totalEdges;
  const edgeStd = Math.sqrt(edgeStrengths.reduce((a, b) => a + (b - meanEdge) ** 2, 0) / totalEdges);
  const edgeCV = edgeStd / (meanEdge || 1);

  // Low edge variation = AI, high = natural
  // Tightened thresholds
  const edgeScore = edgeCV < 0.6 ? 80 :
    edgeCV < 0.9 ? 60 :
    edgeCV < 1.2 ? 40 :
    edgeCV < 1.5 ? 20 : 10;

  return { edgeScore, edgeCoherence: edgeCV, dirEntropy };
}

// ===== Enhanced Skin Analysis with Face Regions =====
function analyzeSkinWithFaces(canvas, ctx, faces) {
  const w = canvas.width, h = canvas.height;
  const data = ctx.getImageData(0, 0, w, h).data;
  const skinMask = new Uint8Array(w * h);
  let skinCount = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    if (r > 95 && g > 40 && b > 20 && r > g && r > b && (r - g) > 15 && r - Math.min(g, b) > 15) {
      skinMask[i / 4] = 1;
      skinCount++;
    }
  }

  const totalPixels = w * h;
  const skinRatio = skinCount / totalPixels;

  if (skinRatio < 0.02) {
    return { hasSkin: false, skinRatio, avgSmoothness: 0, isSmoothed: false, visualization: ctx.getImageData(0, 0, w, h), faceSmoothing: [] };
  }

  const vizData = ctx.createImageData(w, h);
  let smoothnessSum = 0, smoothnessCount = 0;
  const faceSmoothing = [];

  // If we have face regions, analyze those specifically
  if (faces && faces.length > 0) {
    for (const face of faces) {
      const fx = Math.max(1, Math.floor(face.x));
      const fy = Math.max(1, Math.floor(face.y));
      const fw = Math.min(Math.floor(face.width), w - fx - 1);
      const fh = Math.min(Math.floor(face.height), h - fy - 1);

      // Analyze specific face regions: forehead (top 30%), cheeks (middle 40%), chin (bottom 30%)
      const regions = [
        { name: '额头', y1: fy, y2: fy + Math.floor(fh * 0.3), x1: fx + Math.floor(fw * 0.2), x2: fx + Math.floor(fw * 0.8) },
        { name: '左脸颊', y1: fy + Math.floor(fh * 0.3), y2: fy + Math.floor(fh * 0.7), x1: fx, x2: fx + Math.floor(fw * 0.35) },
        { name: '右脸颊', y1: fy + Math.floor(fh * 0.3), y2: fy + Math.floor(fh * 0.7), x1: fx + Math.floor(fw * 0.65), x2: fx + fw },
        { name: '下巴', y1: fy + Math.floor(fh * 0.75), y2: fy + fh, x1: fx + Math.floor(fw * 0.2), x2: fx + Math.floor(fw * 0.8) }
      ];

      const regionResults = [];
      for (const region of regions) {
        let lapSum = 0, lapCount = 0;
        for (let y = Math.max(1, region.y1); y < Math.min(h - 1, region.y2); y++) {
          for (let x = Math.max(1, region.x1); x < Math.min(w - 1, region.x2); x++) {
            if (!skinMask[y * w + x]) continue;
            const getGray = (px) => (data[px * 4] + data[px * 4 + 1] + data[px * 4 + 2]) / 3;
            const idx = y * w + x;
            const lap = Math.abs(getGray(idx - 1) + getGray(idx + 1) + getGray(idx - w) + getGray(idx + w) - 4 * getGray(idx));
            lapSum += lap;
            lapCount++;
          }
        }
        const avgLap = lapCount > 0 ? lapSum / lapCount : 10;
        regionResults.push({ name: region.name, smoothness: avgLap, isSmoothed: avgLap < 6, pixelCount: lapCount });
      }

      faceSmoothing.push({ regions: regionResults });
    }
  }

  // Full image skin analysis for visualization
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const pi = idx * 4;
      if (!skinMask[idx]) {
        vizData.data[pi] = data[pi] * 0.3;
        vizData.data[pi + 1] = data[pi + 1] * 0.3;
        vizData.data[pi + 2] = data[pi + 2] * 0.3;
        vizData.data[pi + 3] = 255;
        continue;
      }

      const getGray = (px) => (data[px * 4] + data[px * 4 + 1] + data[px * 4 + 2]) / 3;
      const lap = Math.abs(getGray(idx - 1) + getGray(idx + 1) + getGray(idx - w) + getGray(idx + w) - 4 * getGray(idx));
      smoothnessSum += lap;
      smoothnessCount++;

      // Color skin heatmap: red = overly smooth (suspicious), green = natural texture
      if (lap < 3) {
        vizData.data[pi] = 220; vizData.data[pi + 1] = 40; vizData.data[pi + 2] = 40;
      } else if (lap < 6) {
        vizData.data[pi] = 255; vizData.data[pi + 1] = 140; vizData.data[pi + 2] = 0;
      } else if (lap < 10) {
        vizData.data[pi] = 180; vizData.data[pi + 1] = 220; vizData.data[pi + 2] = 40;
      } else {
        vizData.data[pi] = 40; vizData.data[pi + 1] = 200; vizData.data[pi + 2] = 40;
      }
      vizData.data[pi + 3] = 255;
    }
  }

  const avgSmoothness = smoothnessCount > 0 ? smoothnessSum / smoothnessCount : 0;
  return {
    hasSkin: true,
    skinRatio,
    avgSmoothness,
    isSmoothed: avgSmoothness < 6,
    visualization: vizData,
    faceSmoothing
  };
}

// ===== Stamp Original Image on Page =====
function stampOriginalImage(imageUrl, overallScore, aiProbability, skinResult) {
  // Find all matching images on the page
  const images = document.querySelectorAll('img');
  for (const img of images) {
    if (!img.src || (!img.src.includes(imageUrl) && imageUrl !== img.src)) continue;
    // Skip already stamped
    if (img.parentElement && img.parentElement.querySelector('.rc-stamp')) continue;

    // Ensure parent is positioned
    const parent = img.parentElement;
    if (parent) {
      const pos = getComputedStyle(parent).position;
      if (pos === 'static') parent.style.position = 'relative';
    }

    // Determine stamp type
    let stampText = '';

    if (aiProbability >= 50) {
      stampText = 'AI 生成';
    } else if (aiProbability >= 35 || (skinResult.hasSkin && skinResult.isSmoothed)) {
      stampText = skinResult.isSmoothed ? '美颜/修图' : '疑似修图';
    } else if (overallScore < 50) {
      stampText = '疑似修改';
    }

    if (!stampText) continue; // Score is fine, no stamp

    const stamp = document.createElement('div');
    stamp.className = 'rc-stamp';
    stamp.style.cssText = `
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%) rotate(-25deg);
      background: rgba(255, 255, 255, 0.85);
      color: #222;
      font-size: ${Math.max(14, Math.min(img.offsetWidth * 0.08, 36))}px;
      font-weight: 400;
      font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
      padding: 6px 20px;
      border: 1px solid #999;
      border-radius: 0;
      letter-spacing: 4px;
      white-space: nowrap;
      pointer-events: none;
      z-index: 10000;
      opacity: 0;
      animation: rc-stamp-in 0.4s ease forwards;
    `;
    stamp.textContent = stampText;

    // Add score badge
    const badge = document.createElement('div');
    badge.style.cssText = `
      position: absolute;
      top: 8px;
      right: 8px;
      background: rgba(255, 255, 255, 0.9);
      color: #222;
      font-size: ${Math.max(11, Math.min(img.offsetWidth * 0.035, 16))}px;
      font-weight: 400;
      font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
      padding: 3px 8px;
      border: 1px solid #ccc;
      border-radius: 0;
      pointer-events: none;
      z-index: 10001;
    `;
    badge.className = 'rc-stamp';
    badge.textContent = `${overallScore}`;

    if (parent) {
      parent.appendChild(stamp);
      parent.appendChild(badge);
    }
  }

  // Inject animation keyframe if not already
  if (!document.getElementById('rc-stamp-style')) {
    const style = document.createElement('style');
    style.id = 'rc-stamp-style';
    style.textContent = `
      @keyframes rc-stamp-in {
        0% { opacity: 0; }
        100% { opacity: 0.85; }
      }
    `;
    document.head.appendChild(style);
  }
}

// ===== Main Panel =====
function openAnalyzerPanel(imageUrl) {
  const existing = document.getElementById('reality-check-panel');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.id = 'reality-check-panel';
  overlay.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: rgba(0,0,0,0.4); z-index: 2147483647;
    display: flex; justify-content: center; align-items: center;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  `;

  const panel = document.createElement('div');
  panel.style.cssText = `
    background: #fff; color: #222; border-radius: 4px;
    width: 1000px; max-width: 95vw; max-height: 90vh; overflow-y: auto;
    padding: 28px; border: 1px solid #e0e0e0;
    position: relative;
  `;

  panel.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;border-bottom:1px solid #e0e0e0;padding-bottom:16px;">
      <h2 style="margin:0;font-size:16px;color:#222;font-weight:400;letter-spacing:0.5px;">图片打假 <span style="font-size:11px;color:#999;font-weight:300;">v2.2.0</span></h2>
      <button id="rc-close" style="background:none;border:1px solid #e0e0e0;color:#999;font-size:18px;cursor:pointer;padding:2px 8px;font-weight:300;line-height:1;">×</button>
    </div>
    <div id="rc-loading" style="text-align:center;padding:40px;">
      <div style="font-size:14px;color:#999;margin-bottom:16px;">正在分析...</div>
      <div id="rc-loading-text" style="color:#999;font-size:12px;">正在加载 TensorFlow.js...</div>
      <div id="rc-loading-progress" style="margin-top:12px;height:2px;background:#f0f0f0;overflow:hidden;">
        <div id="rc-progress-bar" style="height:100%;width:0%;background:#999;transition:width 0.3s;"></div>
      </div>
    </div>
    <div id="rc-results" style="display:none;">
      <div style="display:flex;gap:24px;flex-wrap:wrap;">
        <div style="flex:1;min-width:320px;">
          <div id="rc-score-section" style="text-align:center;padding:24px;background:#f5f5f5;border:1px solid #e0e0e0;border-radius:4px;margin-bottom:16px;">
            <div style="font-size:10px;color:#999;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">综合真实度评分</div>
            <div id="rc-score" style="font-size:64px;font-weight:300;line-height:1;color:#222;"></div>
            <div id="rc-score-label" style="font-size:13px;margin-top:10px;font-weight:300;"></div>
          </div>
          <div style="background:#fff;border:1px solid #e0e0e0;border-radius:4px;padding:16px;margin-bottom:16px;">
            <h3 style="margin:0 0 14px;font-size:10px;color:#999;text-transform:uppercase;letter-spacing:2px;font-weight:400;">ELA 分析</h3>
            <div id="rc-ela-details"></div>
          </div>
          <div style="background:#fff;border:1px solid #e0e0e0;border-radius:4px;padding:16px;margin-bottom:16px;">
            <h3 style="margin:0 0 14px;font-size:10px;color:#999;text-transform:uppercase;letter-spacing:2px;font-weight:400;">AI 生成检测</h3>
            <div id="rc-ai-score" style="text-align:center;padding:16px;margin-bottom:14px;background:#f5f5f5;border:1px solid #e0e0e0;border-radius:4px;">
              <div style="font-size:10px;color:#999;text-transform:uppercase;letter-spacing:1.5px;">AI 生成概率</div>
              <div id="rc-ai-prob" style="font-size:32px;font-weight:300;margin:8px 0;color:#222;"></div>
              <div id="rc-ai-label" style="font-size:12px;font-weight:300;"></div>
            </div>
            <div id="rc-ai-details"></div>
          </div>
          <div id="rc-skin-details-section" style="background:#fff;border:1px solid #e0e0e0;border-radius:4px;padding:16px;margin-bottom:16px;display:none;">
            <h3 style="margin:0 0 14px;font-size:10px;color:#999;text-transform:uppercase;letter-spacing:2px;font-weight:400;">皮肤平滑度</h3>
            <div id="rc-skin-info"></div>
          </div>
          <div id="rc-face-section" style="background:#fff;border:1px solid #e0e0e0;border-radius:4px;padding:16px;margin-bottom:16px;display:none;">
            <h3 style="margin:0 0 14px;font-size:10px;color:#999;text-transform:uppercase;letter-spacing:2px;font-weight:400;">人脸检测</h3>
            <div id="rc-face-details"></div>
          </div>
        </div>
        <div style="flex:1;min-width:320px;">
          <div style="background:#fff;border:1px solid #e0e0e0;border-radius:4px;padding:16px;margin-bottom:16px;">
            <h3 style="margin:0 0 12px;font-size:10px;color:#999;text-transform:uppercase;letter-spacing:2px;font-weight:400;">原图</h3>
            <canvas id="rc-original" style="width:100%;border-radius:2px;"></canvas>
          </div>
          <div style="background:#fff;border:1px solid #e0e0e0;border-radius:4px;padding:16px;margin-bottom:16px;">
            <h3 style="margin:0 0 12px;font-size:10px;color:#999;text-transform:uppercase;letter-spacing:2px;font-weight:400;">ELA 热力图</h3>
            <canvas id="rc-ela" style="width:100%;border-radius:2px;"></canvas>
          </div>
          <div style="background:#fff;border:1px solid #e0e0e0;border-radius:4px;padding:16px;margin-bottom:16px;">
            <h3 style="margin:0 0 12px;font-size:10px;color:#999;text-transform:uppercase;letter-spacing:2px;font-weight:400;">频谱分析</h3>
            <canvas id="rc-spectrum" style="width:100%;border-radius:2px;image-rendering:pixelated;"></canvas>
          </div>
          <div id="rc-skin-section" style="background:#fff;border:1px solid #e0e0e0;border-radius:4px;padding:16px;display:none;">
            <h3 style="margin:0 0 12px;font-size:10px;color:#999;text-transform:uppercase;letter-spacing:2px;font-weight:400;">皮肤分析热力图</h3>
            <canvas id="rc-skin" style="width:100%;border-radius:2px;"></canvas>
          </div>
        </div>
      </div>
    </div>
  `;

  overlay.appendChild(panel);
  document.body.appendChild(overlay);

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) overlay.remove();
  });
  panel.querySelector('#rc-close').addEventListener('click', () => overlay.remove());

  analyzeImage(imageUrl, panel);
}

function setProgress(panel, pct, text) {
  const bar = panel.querySelector('#rc-progress-bar');
  const txt = panel.querySelector('#rc-loading-text');
  if (bar) bar.style.width = pct + '%';
  if (txt) txt.textContent = text;
}

async function analyzeImage(imageUrl, panel) {
  try {
    setProgress(panel, 5, '正在加载图片...');
    const img = await loadImage(imageUrl);
    const w = img.width || img.naturalWidth;
    const h = img.height || img.naturalHeight;

    // Setup canvases
    const origCanvas = panel.querySelector('#rc-original');
    origCanvas.width = w;
    origCanvas.height = h;
    const origCtx = origCanvas.getContext('2d');
    origCtx.drawImage(img, 0, 0);

    // ELA
    setProgress(panel, 15, '正在进行 ELA 分析...');
    const elaResult = await performELA(origCanvas, origCtx, img);
    const elaCanvas = panel.querySelector('#rc-ela');
    elaCanvas.width = w;
    elaCanvas.height = h;
    elaCanvas.getContext('2d').putImageData(elaResult.heatmap, 0, 0);

    // Face detection
    setProgress(panel, 30, '正在检测人脸...');
    let faces = [];
    try {
      faces = await detectFacesBlazeFace(origCanvas);
    } catch (e) {
      console.warn('BlazeFace failed, using heuristic:', e);
    }
    if (faces.length === 0) {
      faces = detectFacesHeuristic(origCanvas, origCtx);
    }

    // Draw face boxes on original
    if (faces.length > 0) {
      const faceCtx = origCanvas.getContext('2d');
      faceCtx.strokeStyle = '#999';
      faceCtx.lineWidth = 1;
      for (const face of faces) {
        faceCtx.strokeRect(face.x, face.y, face.width, face.height);
      }
    }

    // Skin analysis with face regions
    setProgress(panel, 45, '正在分析皮肤平滑度...');
    const skinResult = analyzeSkinWithFaces(origCanvas, origCtx, faces);

    // AI-generated image detection
    setProgress(panel, 55, '正在进行频谱分析...');
    const freqResult = analyzeFrequencyDomain(origCanvas, origCtx);

    setProgress(panel, 65, '正在分析噪点模式...');
    const noiseResult = analyzeNoisePattern(origCanvas, origCtx);

    setProgress(panel, 72, '正在分析色彩分布...');
    const colorResult = analyzeColorHistogram(origCanvas, origCtx);

    setProgress(panel, 80, '正在分析纹理规律性...');
    const textureResult = analyzeTextureRegularity(origCanvas, origCtx);

    setProgress(panel, 87, '正在分析对称性...');
    const symmetryResult = analyzeSymmetry(origCanvas, origCtx, faces);

    setProgress(panel, 92, '正在分析边缘一致性...');
    const edgeResult = analyzeEdgeCoherence(origCanvas, origCtx);

    setProgress(panel, 97, '正在计算综合评分...');

    // Calculate AI probability
    const aiProbability = Math.round(
      freqResult.freqScore * 0.20 +
      noiseResult.noiseScore * 0.20 +
      colorResult.colorScore * 0.15 +
      textureResult.textureScore * 0.15 +
      symmetryResult.symmetryScore * 0.15 +
      edgeResult.edgeScore * 0.15
    );

    // Screenshot detection
    const ssResult = detectScreenshot(origCanvas, origCtx);

    // Calculate overall scores
    const scores = calculateScore(elaResult, skinResult, aiProbability);

    // Don't stamp screenshots
    if (!ssResult.isScreenshot) {
      stampOriginalImage(imageUrl, scores.overall, aiProbability, skinResult);
    }

    // ===== Render Results =====
    const loading = panel.querySelector('#rc-loading');
    const results = panel.querySelector('#rc-results');
    loading.style.display = 'none';
    results.style.display = 'block';

    // Overall score
    const scoreEl = panel.querySelector('#rc-score');
    const scoreLabelEl = panel.querySelector('#rc-score-label');

    if (ssResult.isScreenshot) {
      scoreEl.textContent = '—';
      scoreEl.style.color = '#999';
      scoreLabelEl.textContent = '截图（非拍摄照片，评分不适用）';
      scoreLabelEl.style.color = '#999';
    } else {
      scoreEl.textContent = scores.overall;
      if (scores.overall >= 75) {
        scoreEl.style.color = '#222';
        scoreLabelEl.textContent = '真实';
        scoreLabelEl.style.color = '#666';
      } else if (scores.overall >= 40) {
        scoreEl.style.color = '#222';
        scoreLabelEl.textContent = '可疑';
        scoreLabelEl.style.color = '#666';
      } else {
        scoreEl.style.color = '#222';
        scoreLabelEl.textContent = '虚假';
        scoreLabelEl.style.color = '#666';
      }
    }

    // ELA details (high = good → green)
    panel.querySelector('#rc-ela-details').innerHTML = `
      ${makeBarGood('ELA 一致性', scores.elaScore, 'ELA分数越高表示压缩越均匀')}
      ${makeBarGood('纹理自然度', scores.textureScore, '纹理方差越高越自然')}
      ${makeBarGood('噪点一致性', scores.noiseScore, '噪点分布是否均匀')}
    `;

    // AI detection details
    const aiProb = panel.querySelector('#rc-ai-prob');
    const aiLabel = panel.querySelector('#rc-ai-label');
    aiProb.textContent = aiProbability + '%';
    if (aiProbability >= 60) {
      aiProb.style.color = '#222';
      aiLabel.textContent = '高度疑似AI生成';
      aiLabel.style.color = '#666';
    } else if (aiProbability >= 35) {
      aiProb.style.color = '#222';
      aiLabel.textContent = '可能含AI生成成分';
      aiLabel.style.color = '#666';
    } else {
      aiProb.style.color = '#222';
      aiLabel.textContent = '较可能为真实拍摄';
      aiLabel.style.color = '#999';
    }

    panel.querySelector('#rc-ai-details').innerHTML = `
      ${makeBar('频谱特征', freqResult.freqScore, '高频成分比例: ' + (freqResult.highFreqRatio * 100).toFixed(1) + '%')}
      ${makeBar('噪点均匀度', noiseResult.noiseScore, '变异系数: ' + noiseResult.coeffOfVariation.toFixed(3))}
      ${makeBar('色彩分布', colorResult.colorScore, '直方图平滑度: ' + colorResult.avgSmooth.toFixed(5))}
      ${makeBar('纹理规律性', textureResult.textureScore, 'LBP熵: ' + textureResult.lbpEntropy.toFixed(2))}
      ${makeBar('面部对称性', symmetryResult.symmetryScore, '对称差异: ' + symmetryResult.symmetryValue.toFixed(1))}
      ${makeBar('边缘一致性', edgeResult.edgeScore, '变异系数: ' + edgeResult.edgeCoherence.toFixed(3))}
    `;

    // Spectrum visualization
    const specCanvas = panel.querySelector('#rc-spectrum');
    specCanvas.width = freqResult.spectrumCanvas.width;
    specCanvas.height = freqResult.spectrumCanvas.height;
    specCanvas.getContext('2d').drawImage(freqResult.spectrumCanvas, 0, 0);

    // Skin section
    if (skinResult.hasSkin) {
      panel.querySelector('#rc-skin-section').style.display = 'block';
      const skinCanvas = panel.querySelector('#rc-skin');
      skinCanvas.width = w;
      skinCanvas.height = h;
      skinCanvas.getContext('2d').putImageData(skinResult.visualization, 0, 0);

      const skinInfoSection = panel.querySelector('#rc-skin-details-section');
      skinInfoSection.style.display = 'block';
      let skinHtml = `
        ${makeBarGood('皮肤自然度', scores.skinScore, '平均平滑度: ' + skinResult.avgSmoothness.toFixed(1))}
        <div style="font-size:12px;color:#999;margin-top:8px;">
          肤色像素占比: ${(skinResult.skinRatio * 100).toFixed(1)}% |
          判定: <span style="color:#222;">${skinResult.isSmoothed ? '疑似美颜' : '较自然'}</span>
        </div>
      `;

      if (skinResult.faceSmoothing && skinResult.faceSmoothing.length > 0) {
        skinHtml += `<div style="margin-top:12px;font-size:12px;">`;
        for (let fi = 0; fi < skinResult.faceSmoothing.length; fi++) {
          skinHtml += `<div style="color:#222;margin-bottom:4px;font-weight:400;">人脸 ${fi + 1} 区域分析:</div>`;
          for (const region of skinResult.faceSmoothing[fi].regions) {
            if (region.pixelCount < 10) continue;
            skinHtml += `<div style="padding:2px 0;color:#666;">  ${region.name}: <span style="color:#222;">${region.smoothness.toFixed(1)}</span> ${region.isSmoothed ? '(过度平滑)' : '(自然)'}</div>`;
          }
        }
        skinHtml += `</div>`;
      }
      panel.querySelector('#rc-skin-info').innerHTML = skinHtml;
    }

    // Face section
    if (faces.length > 0) {
      const faceSection = panel.querySelector('#rc-face-section');
      faceSection.style.display = 'block';
      panel.querySelector('#rc-face-details').innerHTML = `
        <div style="font-size:13px;color:#666;">
          检测到 <span style="color:#222;font-weight:400;">${faces.length}</span> 张人脸
          ${faces.map((f, i) => `<div style="margin-top:4px;">人脸 ${i + 1}: 置信度 ${(f.score * 100).toFixed(0)}%, 大小 ${Math.round(f.width)}×${Math.round(f.height)}px</div>`).join('')}
        </div>
      `;
    }

  } catch (err) {
    panel.querySelector('#rc-loading').innerHTML = `
      <div style="color:#222;font-size:14px;">分析失败</div>
      <div style="color:#666;margin-top:8px;font-size:13px;">${err.message}</div>
      <div style="color:#999;margin-top:4px;font-size:12px;">可能原因：跨域限制、图片格式不支持</div>
    `;
  }
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => {
      const img2 = new Image();
      img2.onload = () => resolve(img2);
      img2.onerror = () => reject(new Error('无法加载图片，可能存在跨域限制'));
      img2.src = url;
    };
    img.src = url;
  });
}

function performELA(canvas, ctx, img) {
  const w = img.width, h = img.height;
  const originalData = ctx.getImageData(0, 0, w, h);

  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = w;
  tempCanvas.height = h;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(img, 0, 0);
  const jpegDataUrl = tempCanvas.toDataURL('image/jpeg', 0.95);

  const recompImg = new Image();
  return new Promise((resolve) => {
    recompImg.onload = () => {
      tempCtx.drawImage(recompImg, 0, 0);
      const recompData = tempCtx.getImageData(0, 0, w, h);
      const diffData = ctx.createImageData(w, h);
      let totalDiff = 0;
      const blockSize = 16;
      const blocksX = Math.ceil(w / blockSize);
      const blocksY = Math.ceil(h / blockSize);
      const blockAvgs = [];

      for (let i = 0; i < originalData.data.length; i += 4) {
        const dr = Math.abs(originalData.data[i] - recompData.data[i]);
        const dg = Math.abs(originalData.data[i + 1] - recompData.data[i + 1]);
        const db = Math.abs(originalData.data[i + 2] - recompData.data[i + 2]);
        const diff = (dr + dg + db) / 3;
        totalDiff += diff;

        // Color ELA heatmap: blue→green→yellow→red for low→high difference
        const scaled = Math.min(255, diff * 10);
        const t = scaled / 255;
        let r, g, b;
        if (t < 0.25) { r = 0; g = Math.round(t * 4 * 255); b = 255; }
        else if (t < 0.5) { r = 0; g = 255; b = Math.round((1 - (t - 0.25) * 4) * 255); }
        else if (t < 0.75) { r = Math.round((t - 0.5) * 4 * 255); g = 255; b = 0; }
        else { r = 255; g = Math.round((1 - (t - 0.75) * 4) * 255); b = 0; }
        diffData.data[i] = r;
        diffData.data[i + 1] = g;
        diffData.data[i + 2] = b;
        diffData.data[i + 3] = 255;
      }

      const pixelCount = w * h;
      const avgDiff = totalDiff / pixelCount;

      for (let by = 0; by < blocksY; by++) {
        for (let bx = 0; bx < blocksX; bx++) {
          let sum = 0, count = 0;
          for (let y = by * blockSize; y < Math.min((by + 1) * blockSize, h); y++) {
            for (let x = bx * blockSize; x < Math.min((bx + 1) * blockSize, w); x++) {
              const idx = (y * w + x) * 4;
              sum += (Math.abs(originalData.data[idx] - recompData.data[idx]) +
                Math.abs(originalData.data[idx + 1] - recompData.data[idx + 1]) +
                Math.abs(originalData.data[idx + 2] - recompData.data[idx + 2])) / 3;
              count++;
            }
          }
          blockAvgs.push(sum / count);
        }
      }

      const blockMean = blockAvgs.reduce((a, b) => a + b, 0) / blockAvgs.length;
      const blockVariance = blockAvgs.reduce((a, b) => a + (b - blockMean) ** 2, 0) / blockAvgs.length;

      resolve({ heatmap: diffData, avgDiff, blockVariance, blockMean });
    };
    recompImg.src = jpegDataUrl;
  });
}

function calculateScore(elaResult, skinResult, aiProbability) {
  const elaScore = Math.max(0, Math.min(100, 100 - elaResult.blockVariance * 2));
  const textureScore = Math.max(0, Math.min(100,
    elaResult.avgDiff < 1 ? 30 : elaResult.avgDiff < 5 ? 80 : elaResult.avgDiff < 15 ? 60 : 40
  ));
  const noiseScore = Math.max(0, Math.min(100, 100 - Math.sqrt(elaResult.blockVariance) * 5));

  let skinScore = 70;
  if (skinResult.hasSkin) {
    skinScore = Math.max(0, Math.min(100,
      skinResult.avgSmoothness < 2 ? 10 :
      skinResult.avgSmoothness < 4 ? 25 :
      skinResult.avgSmoothness < 6 ? 40 :
      skinResult.avgSmoothness < 8 ? 60 :
      skinResult.avgSmoothness < 12 ? 75 : 90
    ));
  }

  // Factor in AI probability (inverted: high AI prob = low reality score)
  const aiRealityScore = 100 - aiProbability;

  // Use minimum of component scores to prevent any single bad signal from being averaged away
  // AI probability has highest weight; skin smoothness is critical for beauty filter detection
  let overall;
  if (skinResult.hasSkin) {
    const weighted = Math.round(elaScore * 0.15 + textureScore * 0.05 + noiseScore * 0.05 + skinScore * 0.30 + aiRealityScore * 0.45);
    // Cap overall score: if AI probability > 50% or skin is heavily smoothed, cap at 50
    const cap = (aiProbability > 50 || skinScore < 30) ? 45 : (aiProbability > 35 || skinScore < 45) ? 60 : 100;
    overall = Math.min(weighted, cap);
  } else {
    const weighted = Math.round(elaScore * 0.2 + textureScore * 0.05 + noiseScore * 0.05 + aiRealityScore * 0.7);
    const cap = aiProbability > 50 ? 40 : aiProbability > 35 ? 55 : 100;
    overall = Math.min(weighted, cap);
  }

  return {
    overall: Math.max(0, Math.min(100, overall)),
    elaScore: Math.round(elaScore),
    textureScore: Math.round(textureScore),
    noiseScore: Math.round(noiseScore),
    skinScore: Math.round(skinScore)
  };
}

function makeBarGood(label, value, desc) {
  // Higher = better, darker fill for higher values
  const gray = Math.round(180 - value * 1.4); // 180 (light) at 0 → 40 (dark) at 100
  const fillColor = `rgb(${gray},${gray},${gray})`;
  return `
    <div style="margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px;font-weight:300;">
        <span style="color:#444;">${label}</span>
        <span style="color:#222;font-weight:500;">${value}</span>
      </div>
      <div style="height:2px;background:#e0e0e0;overflow:hidden;">
        <div style="height:100%;width:${value}%;background:${fillColor};transition:width 0.5s;"></div>
      </div>
      <div style="font-size:11px;color:#999;margin-top:4px;">${desc}</div>
    </div>
  `;
}

function makeBar(label, value, desc) {
  // Higher = worse (more suspicious), darker fill for higher values
  const gray = Math.round(180 - value * 1.4);
  const fillColor = `rgb(${gray},${gray},${gray})`;
  return `
    <div style="margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px;font-weight:300;">
        <span style="color:#444;">${label}</span>
        <span style="color:#222;font-weight:500;">${value}</span>
      </div>
      <div style="height:2px;background:#e0e0e0;overflow:hidden;">
        <div style="height:100%;width:${value}%;background:${fillColor};transition:width 0.5s;"></div>
      </div>
      <div style="font-size:11px;color:#999;margin-top:4px;">${desc}</div>
    </div>
  `;
}

// ===== Auto-Detection System =====
let autoDetectEnabled = false;
let faceOnlyMode = true;
let autoScanObserver = null;
let autoScanQueue = [];
let autoScanProcessing = 0;
const AUTO_SCAN_CONCURRENCY = 2;
const MIN_IMG_SIZE = 100;

// Initialize settings from storage
chrome.storage.sync.get({ autoDetect: true, faceOnly: true }, (data) => {
  autoDetectEnabled = data.autoDetect;
  faceOnlyMode = data.faceOnly;
  if (autoDetectEnabled) startAutoScan();
});

// Listen for toggle changes in real-time
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'sync') {
    if (changes.autoDetect) {
      autoDetectEnabled = changes.autoDetect.newValue;
      if (autoDetectEnabled) {
        startAutoScan();
      } else {
        stopAutoScan();
      }
    }
    if (changes.faceOnly) {
      faceOnlyMode = changes.faceOnly.newValue;
    }
  }
});

// Listen for messages from popup/background
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'RC_START_AUTO_SCAN') {
    autoDetectEnabled = true;
    startAutoScan();
  } else if (message.type === 'RC_STOP_AUTO_SCAN') {
    autoDetectEnabled = false;
    stopAutoScan();
  } else if (message.type === 'RC_UPDATE_FACE_ONLY') {
    faceOnlyMode = message.faceOnly;
  }
});

function startAutoScan() {
  // Scan existing images
  scanPageImages();
  // Observe new images
  if (!autoScanObserver) {
    autoScanObserver = new MutationObserver((mutations) => {
      if (!autoDetectEnabled) return;
      let hasNewImages = false;
      for (const m of mutations) {
        for (const node of m.addedNodes) {
          if (node.nodeName === 'IMG') { hasNewImages = true; break; }
          if (node.querySelectorAll) {
            const imgs = node.querySelectorAll('img');
            if (imgs.length > 0) { hasNewImages = true; break; }
          }
        }
        if (hasNewImages) break;
      }
      if (hasNewImages) scanPageImages();
    });
    autoScanObserver.observe(document.body, { childList: true, subtree: true });
  }
}

function stopAutoScan() {
  // Stop observer
  if (autoScanObserver) {
    autoScanObserver.disconnect();
    autoScanObserver = null;
  }
  // Clear queue
  autoScanQueue = [];
  // Remove all stamps
  document.querySelectorAll('.rc-stamp').forEach(el => el.remove());
}

function scanPageImages() {
  if (!autoDetectEnabled) return;
  const images = document.querySelectorAll('img');
  for (const img of images) {
    // Skip small images (icons/thumbnails)
    if (img.naturalWidth <= MIN_IMG_SIZE || img.naturalHeight <= MIN_IMG_SIZE) continue;
    if (img.width <= MIN_IMG_SIZE || img.height <= MIN_IMG_SIZE) continue;
    // Skip already stamped
    if (img.parentElement && img.parentElement.querySelector('.rc-stamp')) continue;
    // Skip already queued (mark with data attribute)
    if (img.dataset.rcQueued) continue;
    img.dataset.rcQueued = '1';
    autoScanQueue.push(img);
  }
  processAutoScanQueue();
}

function processAutoScanQueue() {
  while (autoScanProcessing < AUTO_SCAN_CONCURRENCY && autoScanQueue.length > 0) {
    const img = autoScanQueue.shift();
    if (!autoDetectEnabled) return;
    // Verify image is still in DOM and not stamped
    if (!document.contains(img)) continue;
    if (img.parentElement && img.parentElement.querySelector('.rc-stamp')) continue;
    autoScanProcessing++;
    autoAnalyzeImage(img).finally(() => {
      autoScanProcessing--;
      if (autoDetectEnabled) processAutoScanQueue();
    });
  }
}

function addProgressBar(imgEl) {
  const parent = imgEl.parentElement;
  if (!parent) return null;
  const pos = getComputedStyle(parent).position;
  if (pos === 'static') parent.style.position = 'relative';

  const bar = document.createElement('div');
  bar.className = 'rc-progress';
  bar.style.cssText = `
    position: absolute; bottom: 0; left: 0; width: 100%; height: 3px;
    background: rgba(0,0,0,0.15); z-index: 10000; pointer-events: none;
    overflow: hidden;
  `;
  const fill = document.createElement('div');
  fill.className = 'rc-progress-fill';
  fill.style.cssText = `
    height: 100%; width: 0%; background: #888;
    transition: width 0.3s ease;
  `;
  bar.appendChild(fill);
  parent.appendChild(bar);
  return fill;
}

function updateProgress(fill, pct) {
  if (fill) fill.style.width = pct + '%';
}

function removeProgressBar(imgEl) {
  const parent = imgEl.parentElement;
  if (!parent) return;
  const bar = parent.querySelector('.rc-progress');
  if (bar) bar.remove();
}

async function autoAnalyzeImage(imgEl) {
  const progressFill = addProgressBar(imgEl);
  try {
    const url = imgEl.src;
    if (!url) { removeProgressBar(imgEl); return; }

    updateProgress(progressFill, 5);
    const img = await loadImage(url);
    const w = img.width || img.naturalWidth;
    const h = img.height || img.naturalHeight;
    if (w <= MIN_IMG_SIZE || h <= MIN_IMG_SIZE) { removeProgressBar(imgEl); return; }

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    updateProgress(progressFill, 10);
    // Screenshot detection: skip screenshots entirely
    const ssResult = detectScreenshot(canvas, ctx);
    if (ssResult.isScreenshot) {
      removeProgressBar(imgEl);
      return;
    }

    updateProgress(progressFill, 15);
    // Quick face check first when faceOnly mode is on
    const faces = detectFacesHeuristic(canvas, ctx);
    if (faceOnlyMode && faces.length === 0) {
      removeProgressBar(imgEl);
      return;
    }
    updateProgress(progressFill, 20);
    const elaResult = await performELA(canvas, ctx, img);
    updateProgress(progressFill, 40);
    const skinResult = analyzeSkinWithFaces(canvas, ctx, faces);
    updateProgress(progressFill, 50);
    const freqResult = analyzeFrequencyDomain(canvas, ctx);
    updateProgress(progressFill, 60);
    const noiseResult = analyzeNoisePattern(canvas, ctx);
    updateProgress(progressFill, 70);
    const colorResult = analyzeColorHistogram(canvas, ctx);
    updateProgress(progressFill, 78);
    const textureResult = analyzeTextureRegularity(canvas, ctx);
    updateProgress(progressFill, 85);
    const symmetryResult = analyzeSymmetry(canvas, ctx, faces);
    updateProgress(progressFill, 92);
    const edgeResult = analyzeEdgeCoherence(canvas, ctx);
    updateProgress(progressFill, 98);

    const aiProbability = Math.round(
      freqResult.freqScore * 0.20 +
      noiseResult.noiseScore * 0.20 +
      colorResult.colorScore * 0.15 +
      textureResult.textureScore * 0.15 +
      symmetryResult.symmetryScore * 0.15 +
      edgeResult.edgeScore * 0.15
    );

    const scores = calculateScore(elaResult, skinResult, aiProbability);

    updateProgress(progressFill, 100);
    // Brief pause to show 100% then remove
    await new Promise(r => setTimeout(r, 200));
    removeProgressBar(imgEl);

    if (!autoDetectEnabled) return;

    // Stamp the image
    stampOriginalImage(url, scores.overall, aiProbability, skinResult);
  } catch (e) {
    removeProgressBar(imgEl);
    // Silently skip failed images (CORS, etc.)
  }
}
