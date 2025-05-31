package cv

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"sort"
	"strings"

	"gocv.io/x/gocv"
)

type FishBiometric struct {
	ID                  string    `json:"id"`
	Confidence          float64   `json:"confidence"`
	UniqueMarkings      []string  `json:"unique_markings"`
	ScarsAndDamage      []string  `json:"scars_and_damage"`
	DistinctivePatterns []string  `json:"distinctive_patterns"`
	PermanentFeatures   []string  `json:"permanent_features"`
	NormalizedFinShape  []float64 `json:"normalized_fin_shape"`
	NormalizedBodyRatio float64   `json:"normalized_body_ratio"`
	BiometricSignature  string    `json:"biometric_signature"`
}

type FishBodyGeometry struct {
	centerX, centerY float64
	majorAxis        float64 // Length of the fish body
	minorAxis        float64 // Width of the fish body
	orientation      float64 // Angle of the major axis
	boundingRect     image.Rectangle
}

type FishBiometricAnalyzer struct {
	detector gocv.SIFT
}

func NewFishBiometricAnalyzer() *FishBiometricAnalyzer {
	return &FishBiometricAnalyzer{
		detector: gocv.NewSIFT(),
	}
}

func (fba *FishBiometricAnalyzer) Close() {
	fba.detector.Close()
}

func (fba *FishBiometricAnalyzer) IdentifyIndividualFish(imagePath string) (*FishBiometric, error) {
	// Load and preprocess image
	img := gocv.IMRead(imagePath, gocv.IMReadColor)
	if img.Empty() {
		return nil, fmt.Errorf("could not load image: %s", imagePath)
	}
	defer img.Close()

	// Normalize image for consistent analysis
	normalizedImg := fba.normalizeImage(img)
	defer normalizedImg.Close()

	// Extract fish region
	fishContour, fishMask := fba.extractFishRegion(normalizedImg)
	defer fishMask.Close()

	if len(fishContour) < 20 {
		return nil, fmt.Errorf("insufficient fish detail detected for individual identification")
	}

	// Calculate fish body geometry for transformation-invariant features
	bodyGeometry := fba.calculateBodyGeometry(fishContour)

	// Extract transformation-invariant biometric features
	uniqueMarkings := fba.detectUniqueMarkingsTransformInvariant(normalizedImg, fishMask, bodyGeometry)
	scarsAndDamage := fba.detectScarsAndDamageTransformInvariant(normalizedImg, fishMask, bodyGeometry)
	distinctivePatterns := fba.extractDistinctivePatterns(normalizedImg, fishMask)
	permanentFeatures := fba.analyzePermanentFeatures(normalizedImg, fishMask, fishContour)
	normalizedFinShape := fba.extractNormalizedFinShape(fishMask, fishContour)
	normalizedBodyRatio := fba.calculateNormalizedBodyRatio(fishContour)

	// Create biometric signature focusing on permanent, unique characteristics
	biometricSignature := fba.createBiometricSignature(
		uniqueMarkings, scarsAndDamage, distinctivePatterns,
		permanentFeatures, normalizedFinShape, normalizedBodyRatio,
	)

	// Calculate confidence based on number of distinctive features found
	confidence := fba.calculateBiometricConfidence(
		uniqueMarkings, scarsAndDamage, distinctivePatterns, permanentFeatures,
	)

	// Generate individual fish ID from biometric signature
	id := fba.generateIndividualFishID(biometricSignature)

	return &FishBiometric{
		ID:                  id,
		Confidence:          confidence,
		UniqueMarkings:      uniqueMarkings,
		ScarsAndDamage:      scarsAndDamage,
		DistinctivePatterns: distinctivePatterns,
		PermanentFeatures:   permanentFeatures,
		NormalizedFinShape:  normalizedFinShape,
		NormalizedBodyRatio: normalizedBodyRatio,
		BiometricSignature:  biometricSignature,
	}, nil
}

func (fba *FishBiometricAnalyzer) normalizeImage(img gocv.Mat) gocv.Mat {
	// Convert to LAB color space for better lighting normalization
	lab := gocv.NewMat()
	gocv.CvtColor(img, &lab, gocv.ColorBGRToLab)

	// Split channels
	channels := gocv.Split(lab)
	defer func() {
		for _, ch := range channels {
			ch.Close()
		}
	}()

	// Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
	clahe := gocv.NewCLAHE()
	defer clahe.Close()

	clahe.Apply(channels[0], &channels[0])

	// Merge channels back
	normalized := gocv.NewMat()
	gocv.Merge(channels, &normalized)

	// Convert back to BGR
	result := gocv.NewMat()
	gocv.CvtColor(normalized, &result, gocv.ColorLabToBGR)
	normalized.Close()
	lab.Close()

	return result
}

func (fba *FishBiometricAnalyzer) extractFishRegion(img gocv.Mat) ([]image.Point, gocv.Mat) {
	// Convert to grayscale
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)

	// Apply multiple edge detection methods for robust contour detection
	edges1 := gocv.NewMat()
	defer edges1.Close()
	gocv.Canny(gray, &edges1, 50, 150)

	// Morphological operations to connect edges
	kernel := gocv.GetStructuringElement(gocv.MorphEllipse, image.Pt(3, 3))
	defer kernel.Close()

	edges2 := gocv.NewMat()
	defer edges2.Close()
	gocv.MorphologyEx(edges1, &edges2, gocv.MorphClose, kernel)

	// Find contours
	contours := gocv.FindContours(edges2, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	// Find the most fish-like contour (largest with reasonable aspect ratio)
	var bestContour []image.Point
	maxScore := 0.0

	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)

		if area > 1000 { // Minimum size threshold
			points := gocv.NewPointVectorFromPoints(contour.ToPoints())
			rect := gocv.BoundingRect(points)
			points.Close()

			aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

			// Fish typically have aspect ratio between 1.5 and 4.0
			if aspectRatio >= 1.2 && aspectRatio <= 5.0 {
				score := area * (1.0 / math.Abs(aspectRatio-2.5)) // Prefer ~2.5 aspect ratio
				if score > maxScore {
					maxScore = score
					bestContour = contour.ToPoints()
				}
			}
		}
	}

	// Create mask
	mask := gocv.NewMatWithSize(img.Rows(), img.Cols(), gocv.MatTypeCV8UC1)
	mask.SetTo(gocv.NewScalar(0, 0, 0, 0))

	if len(bestContour) > 0 {
		pointsVector := gocv.NewPointsVectorFromPoints([][]image.Point{bestContour})
		defer pointsVector.Close()
		gocv.FillPoly(&mask, pointsVector, color.RGBA{255, 255, 255, 255})
	}

	return bestContour, mask
}

func (fba *FishBiometricAnalyzer) calculateBodyGeometry(contour []image.Point) FishBodyGeometry {
	// Convert to PointVector for OpenCV operations
	points := gocv.NewPointVectorFromPoints(contour)
	defer points.Close()

	// Get bounding rectangle
	rect := gocv.BoundingRect(points)

	// Calculate center of mass (geometric center)
	sumX, sumY := 0, 0
	for _, pt := range contour {
		sumX += pt.X
		sumY += pt.Y
	}
	centerX := float64(sumX) / float64(len(contour))
	centerY := float64(sumY) / float64(len(contour))

	// Fit ellipse to get major/minor axes and orientation
	// For simplicity, use bounding rectangle dimensions as approximation
	majorAxis := math.Max(float64(rect.Dx()), float64(rect.Dy()))
	minorAxis := math.Min(float64(rect.Dx()), float64(rect.Dy()))

	// Calculate orientation using PCA-like approach on contour points
	orientation := fba.calculateContourOrientation(contour, centerX, centerY)

	return FishBodyGeometry{
		centerX:      centerX,
		centerY:      centerY,
		majorAxis:    majorAxis,
		minorAxis:    minorAxis,
		orientation:  orientation,
		boundingRect: rect,
	}
}

func (fba *FishBiometricAnalyzer) calculateContourOrientation(contour []image.Point, centerX, centerY float64) float64 {
	// Calculate covariance matrix for PCA
	var cxx, cyy, cxy float64
	n := float64(len(contour))

	for _, pt := range contour {
		dx := float64(pt.X) - centerX
		dy := float64(pt.Y) - centerY
		cxx += dx * dx
		cyy += dy * dy
		cxy += dx * dy
	}

	cxx /= n
	cyy /= n
	cxy /= n

	// Calculate principal component angle
	if cxx == cyy {
		return 0 // No clear orientation
	}

	orientation := 0.5 * math.Atan2(2*cxy, cxx-cyy)
	return orientation
}

func (fba *FishBiometricAnalyzer) detectUniqueMarkingsTransformInvariant(img gocv.Mat, mask gocv.Mat, geometry FishBodyGeometry) []string {
	// Convert to HSV for better color analysis
	hsv := gocv.NewMat()
	defer hsv.Close()
	gocv.CvtColor(img, &hsv, gocv.ColorBGRToHSV)

	var markings []string

	// Detect various types of markings with fuzzy categorization
	markings = append(markings, fba.detectDarkSpots(hsv, mask, geometry)...)
	markings = append(markings, fba.detectColoredMarkings(hsv, mask, geometry)...)
	markings = append(markings, fba.detectBrightSpots(hsv, mask, geometry)...)

	return markings
}

func (fba *FishBiometricAnalyzer) detectScarsAndDamageTransformInvariant(img gocv.Mat, mask gocv.Mat, geometry FishBodyGeometry) []string {
	// Convert to grayscale for edge detection
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)

	// Apply very conservative edge detection to find only significant disruptions
	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(gray, &edges, 200, 400)

	// Apply mask to focus on fish region
	fishEdges := gocv.NewMat()
	defer fishEdges.Close()
	gocv.BitwiseAnd(edges, mask, &fishEdges)

	var scars []string

	// Find edge contours that might represent scars or damage
	edgeContours := gocv.FindContours(fishEdges, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer edgeContours.Close()

	for i := 0; i < edgeContours.Size(); i++ {
		contour := edgeContours.At(i)
		points := contour.ToPoints()

		if len(points) > 5 {
			points_vec := gocv.NewPointVectorFromPoints(points)
			arcLength := gocv.ArcLength(points_vec, false)
			area := gocv.ContourArea(points_vec)
			rect := gocv.BoundingRect(points_vec)
			points_vec.Close()

			// Look for significant elongated features (potential scars) - ultra conservative
			// Only detect major scars/wounds, completely ignore minor patterns
			if arcLength > 100 && area > 300 && area < 600 {
				// Calculate center position for zone classification
				centerX := float64(rect.Min.X + rect.Dx()/2)
				centerY := float64(rect.Min.Y + rect.Dy()/2)

				// Use fuzzy categorical descriptors
				sizeCategory := fba.getScarSizeCategory(arcLength, geometry)
				positionZone := fba.getPositionZone(centerX, centerY, geometry)
				scarType := fba.getScarTypeFromShape(rect, arcLength, area)

				scar := fmt.Sprintf("%s_%s_scar_%s", sizeCategory, scarType, positionZone)
				scars = append(scars, scar)
			}
		}
	}

	return scars
}

func (fba *FishBiometricAnalyzer) detectUniqueMarkings(img gocv.Mat, mask gocv.Mat) []string {
	// Convert to HSV for better color analysis
	hsv := gocv.NewMat()
	defer hsv.Close()
	gocv.CvtColor(img, &hsv, gocv.ColorBGRToHSV)

	var markings []string

	// Detect dark spots (potential unique markings)
	darkMask := gocv.NewMat()
	defer darkMask.Close()

	// Create threshold mats for dark regions
	lowerBound := gocv.NewMatFromScalar(gocv.NewScalar(0, 0, 0, 0), gocv.MatTypeCV8UC3)
	defer lowerBound.Close()
	upperBound := gocv.NewMatFromScalar(gocv.NewScalar(180, 255, 80, 0), gocv.MatTypeCV8UC3)
	defer upperBound.Close()

	gocv.InRange(hsv, lowerBound, upperBound, &darkMask)

	// Apply fish mask to focus only on fish region
	fishDarkMask := gocv.NewMat()
	defer fishDarkMask.Close()
	gocv.BitwiseAnd(darkMask, mask, &fishDarkMask)

	// Find dark spot contours
	spotContours := gocv.FindContours(fishDarkMask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer spotContours.Close()

	for i := 0; i < spotContours.Size(); i++ {
		contour := spotContours.At(i)
		area := gocv.ContourArea(contour)

		if area > 20 && area < 2000 { // Reasonable spot sizes
			points := gocv.NewPointVectorFromPoints(contour.ToPoints())
			rect := gocv.BoundingRect(points)
			points.Close()

			// Describe spot location and characteristics
			centerX := rect.Min.X + rect.Dx()/2
			centerY := rect.Min.Y + rect.Dy()/2
			marking := fmt.Sprintf("dark_spot_%.0f_%d_%d", area, centerX, centerY)
			markings = append(markings, marking)
		}
	}

	return markings
}

func (fba *FishBiometricAnalyzer) detectScarsAndDamage(img gocv.Mat, mask gocv.Mat) []string {
	// Convert to grayscale for edge detection
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)

	// Apply strong edge detection to find disruptions in fish outline/texture
	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(gray, &edges, 80, 200)

	// Apply mask to focus on fish region
	fishEdges := gocv.NewMat()
	defer fishEdges.Close()
	gocv.BitwiseAnd(edges, mask, &fishEdges)

	var scars []string

	// Find edge contours that might represent scars or damage
	edgeContours := gocv.FindContours(fishEdges, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer edgeContours.Close()

	for i := 0; i < edgeContours.Size(); i++ {
		contour := edgeContours.At(i)
		points := contour.ToPoints()

		if len(points) > 5 {
			points_vec := gocv.NewPointVectorFromPoints(points)
			arcLength := gocv.ArcLength(points_vec, false)
			area := gocv.ContourArea(points_vec)
			points_vec.Close()

			// Look for significant elongated features (potential scars) - ultra conservative
			// Only detect major scars/wounds, completely ignore minor patterns
			if arcLength > 300 && area > 500 && area < 800 {
				// Calculate approximate scar characteristics
				points_vec := gocv.NewPointVectorFromPoints(points)
				rect := gocv.BoundingRect(points_vec)
				points_vec.Close()
				aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

				// Require extremely elongated features for true scars
				if aspectRatio > 8.0 || aspectRatio < 0.125 { // Only severe elongated damage
					centerX := rect.Min.X + rect.Dx()/2
					centerY := rect.Min.Y + rect.Dy()/2
					scar := fmt.Sprintf("scar_%.0f_%.1f_%d_%d", arcLength, aspectRatio, centerX, centerY)
					scars = append(scars, scar)
				}
			}
		}
	}

	return scars
}

func (fba *FishBiometricAnalyzer) extractDistinctivePatterns(img gocv.Mat, mask gocv.Mat) []string {
	var patterns []string

	// Analyze overall color characteristics
	patterns = append(patterns, fba.analyzeOverallColorPattern(img, mask)...)

	// Detect specific pattern types
	patterns = append(patterns, fba.detectSpottedPattern(img, mask)...)
	patterns = append(patterns, fba.detectStripedPattern(img, mask)...)
	patterns = append(patterns, fba.detectTexturePattern(img, mask)...)

	return patterns
}

func (fba *FishBiometricAnalyzer) analyzePermanentFeatures(img gocv.Mat, mask gocv.Mat, contour []image.Point) []string {
	var features []string

	if len(contour) < 10 {
		return features
	}

	// Analyze fin positions and shapes (these are relatively permanent)
	// This is a simplified approach - in practice would be more sophisticated
	points_vec := gocv.NewPointVectorFromPoints(contour)
	defer points_vec.Close()

	// Get convex hull to identify protruding features (fins)
	hull := gocv.NewMat()
	defer hull.Close()
	gocv.ConvexHull(points_vec, &hull, false, false)

	// For now, just use contour area as a feature since hull analysis is complex
	contourArea := gocv.ContourArea(points_vec)
	feature := fmt.Sprintf("contour_area_%.0f", contourArea)
	features = append(features, feature)

	// Analyze contour curvature for distinctive shape features
	if len(contour) > 20 {
		// Sample points along contour for curvature analysis
		sampleStep := len(contour) / 10
		for i := sampleStep; i < len(contour)-sampleStep; i += sampleStep {
			// Calculate local curvature (simplified)
			p1 := contour[i-sampleStep]
			p2 := contour[i]
			p3 := contour[i+sampleStep]

			// Calculate vectors
			v1x, v1y := float64(p2.X-p1.X), float64(p2.Y-p1.Y)
			v2x, v2y := float64(p3.X-p2.X), float64(p3.Y-p2.Y)

			// Calculate angle between vectors (curvature indicator)
			dot := v1x*v2x + v1y*v2y
			mag1 := math.Sqrt(v1x*v1x + v1y*v1y)
			mag2 := math.Sqrt(v2x*v2x + v2y*v2y)

			if mag1 > 0 && mag2 > 0 {
				cosAngle := dot / (mag1 * mag2)
				cosAngle = math.Max(-1, math.Min(1, cosAngle)) // Clamp to valid range
				angle := math.Acos(cosAngle)

				if angle > 0.5 { // Significant curvature
					feature := fmt.Sprintf("curve_%d_%.2f", i, angle)
					features = append(features, feature)
				}
			}
		}
	}

	return features
}

func (fba *FishBiometricAnalyzer) extractNormalizedFinShape(mask gocv.Mat, contour []image.Point) []float64 {
	// This would extract fin shape features normalized for size and rotation
	// Simplified implementation
	var finFeatures []float64

	if len(contour) < 10 {
		return finFeatures
	}

	// Extract basic normalized shape descriptors
	points_vec := gocv.NewPointVectorFromPoints(contour)
	defer points_vec.Close()

	area := gocv.ContourArea(points_vec)
	perimeter := gocv.ArcLength(points_vec, true)

	if perimeter > 0 {
		circularity := 4 * math.Pi * area / (perimeter * perimeter)
		finFeatures = append(finFeatures, circularity)
	}

	// Add more normalized descriptors...
	rect := gocv.BoundingRect(points_vec)
	if rect.Dy() > 0 {
		aspectRatio := float64(rect.Dx()) / float64(rect.Dy())
		finFeatures = append(finFeatures, aspectRatio)
	}

	return finFeatures
}

func (fba *FishBiometricAnalyzer) calculateNormalizedBodyRatio(contour []image.Point) float64 {
	if len(contour) < 5 {
		return 0
	}

	points_vec := gocv.NewPointVectorFromPoints(contour)
	defer points_vec.Close()

	rect := gocv.BoundingRect(points_vec)

	if rect.Dy() == 0 {
		return 0
	}

	return float64(rect.Dx()) / float64(rect.Dy())
}

func (fba *FishBiometricAnalyzer) createBiometricSignature(
	markings, scars, patterns, features []string,
	finShape []float64, bodyRatio float64) string {

	// Combine all distinctive features into a normalized signature
	var allFeatures []string

	// Add categorical features
	allFeatures = append(allFeatures, markings...)
	allFeatures = append(allFeatures, scars...)
	allFeatures = append(allFeatures, patterns...)
	allFeatures = append(allFeatures, features...)

	// Add numerical features as strings
	for _, f := range finShape {
		allFeatures = append(allFeatures, fmt.Sprintf("fin_%.3f", f))
	}
	allFeatures = append(allFeatures, fmt.Sprintf("body_%.3f", bodyRatio))

	// Sort for consistency
	sort.Strings(allFeatures)

	return strings.Join(allFeatures, "|")
}

func (fba *FishBiometricAnalyzer) calculateBiometricConfidence(
	markings, scars, patterns, features []string) float64 {

	confidence := 50.0 // Base confidence

	// Boost confidence for each type of distinctive feature found
	confidence += float64(len(markings)) * 0.10 // Unique markings are very valuable
	confidence += float64(len(scars)) * 0.15    // Scars are extremely distinctive
	confidence += float64(len(patterns)) * 0.5  // Patterns help but are less unique
	confidence += float64(len(features)) * 0.3  // General features provide some value

	// Cap at 100%
	return math.Min(100, confidence)
}

func (fba *FishBiometricAnalyzer) generateIndividualFishID(signature string) string {
	// Parse signature into core features for fuzzy matching
	features := fba.extractCoreFeatures(signature)

	// Create fuzzy ID based on feature buckets instead of exact hash
	return fba.generateFuzzyID(features)
}

// extractCoreFeatures extracts the most stable identifying features from the signature
func (fba *FishBiometricAnalyzer) extractCoreFeatures(signature string) map[string]int {
	features := make(map[string]int)

	// Split signature into components
	parts := strings.Split(signature, "|")

	for _, part := range parts {
		// Count categorical features by type
		if strings.Contains(part, "scar") {
			features["scars"]++

			// Categorize scar types
			if strings.Contains(part, "major") || strings.Contains(part, "severe") {
				features["major_scars"]++
			}
		} else if strings.Contains(part, "dark_spot") {
			features["dark_spots"]++
		} else if strings.Contains(part, "bright_spot") {
			features["bright_spots"]++
		} else if strings.Contains(part, "_accent") {
			features["colored_markings"]++
		} else if strings.Contains(part, "spotted") {
			features["spotted_pattern"] = 1
		} else if strings.Contains(part, "striped") {
			features["striped_pattern"] = 1
		} else if strings.Contains(part, "warm_") {
			features["warm_coloring"] = 1
		} else if strings.Contains(part, "cool_") {
			features["cool_coloring"] = 1
		} else if strings.Contains(part, "vivid_coloring") {
			features["vivid_colors"] = 1
		} else if strings.Contains(part, "high_contrast") {
			features["high_contrast"] = 1
		}
	}

	return features
}

// generateFuzzyID creates an ID based on feature ranges rather than exact values
func (fba *FishBiometricAnalyzer) generateFuzzyID(features map[string]int) string {
	var idComponents []string

	// Count major features using fuzzy buckets
	scarCount := features["scars"]
	darkSpotCount := features["dark_spots"]
	brightSpotCount := features["bright_spots"]
	coloredMarkingCount := features["colored_markings"]
	majorScarCount := features["major_scars"]

	// Use broad buckets for counts
	scarBucket := fba.getCountBucket(scarCount)
	spotBucket := fba.getCountBucket(darkSpotCount + brightSpotCount)
	colorBucket := fba.getCountBucket(coloredMarkingCount)
	majorScarBucket := fba.getCountBucket(majorScarCount)

	// Build ID from categorical features
	idComponents = append(idComponents, fmt.Sprintf("S%d", scarBucket))      // Scars
	idComponents = append(idComponents, fmt.Sprintf("M%d", spotBucket))      // Markings (spots)
	idComponents = append(idComponents, fmt.Sprintf("C%d", colorBucket))     // Colored markings
	idComponents = append(idComponents, fmt.Sprintf("X%d", majorScarBucket)) // Major scars

	// Add pattern indicators
	if features["spotted_pattern"] > 0 {
		idComponents = append(idComponents, "P1") // P for Pattern, 1 for spotted
	} else if features["striped_pattern"] > 0 {
		idComponents = append(idComponents, "P2") // P for Pattern, 2 for striped
	} else {
		idComponents = append(idComponents, "P0") // No distinct pattern
	}

	// Create stable, human-readable ID
	baseID := strings.Join(idComponents, "")

	// Add a simple checksum for uniqueness without being overly sensitive
	//checksum := fba.simpleChecksum(baseID)

	return baseID
}

// getBodyRatioBucket groups similar body ratios together
func (fba *FishBiometricAnalyzer) getBodyRatioBucket(ratio float64) int {
	// Group ratios into buckets of 0.2 width
	if ratio < 1.0 {
		return 1
	}
	if ratio < 1.2 {
		return 2
	}
	if ratio < 1.4 {
		return 3
	}
	if ratio < 1.6 {
		return 4
	}
	if ratio < 1.8 {
		return 5
	}
	if ratio < 2.0 {
		return 6
	}
	return 7 // Very elongated fish
}

// getCountBucket groups feature counts into broad ranges
func (fba *FishBiometricAnalyzer) getCountBucket(count int) int {
	if count == 0 {
		return 0
	}
	if count <= 5 {
		return 1
	}
	if count <= 15 {
		return 2
	}
	if count <= 30 {
		return 3
	}
	if count <= 50 {
		return 4
	}
	if count <= 100 {
		return 5
	}
	return 6 // Many features
}

// getAreaBucket groups areas into size categories
func (fba *FishBiometricAnalyzer) getAreaBucket(area float64) int {
	if area < 1000000 {
		return 1
	} // Small
	if area < 5000000 {
		return 2
	} // Medium-Small
	if area < 10000000 {
		return 3
	} // Medium
	if area < 20000000 {
		return 4
	} // Medium-Large
	if area < 50000000 {
		return 5
	} // Large
	return 6 // Very Large
}

// simpleChecksum creates a simple checksum to add uniqueness without being overly sensitive
func (fba *FishBiometricAnalyzer) simpleChecksum(data string) int {
	sum := 0
	for i, char := range data {
		sum += int(char) * (i + 1)
	}
	return sum % 256
}

// detectDarkSpots detects dark markings using fuzzy categorization
func (fba *FishBiometricAnalyzer) detectDarkSpots(hsv gocv.Mat, mask gocv.Mat, geometry FishBodyGeometry) []string {
	var markings []string

	// Create threshold mats for dark regions
	lowerBound := gocv.NewMatFromScalar(gocv.NewScalar(0, 0, 0, 0), gocv.MatTypeCV8UC3)
	defer lowerBound.Close()
	upperBound := gocv.NewMatFromScalar(gocv.NewScalar(180, 255, 80, 0), gocv.MatTypeCV8UC3)
	defer upperBound.Close()

	darkMask := gocv.NewMat()
	defer darkMask.Close()
	gocv.InRange(hsv, lowerBound, upperBound, &darkMask)

	// Apply fish mask
	fishDarkMask := gocv.NewMat()
	defer fishDarkMask.Close()
	gocv.BitwiseAnd(darkMask, mask, &fishDarkMask)

	// Find contours
	spotContours := gocv.FindContours(fishDarkMask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer spotContours.Close()

	for i := 0; i < spotContours.Size(); i++ {
		contour := spotContours.At(i)
		area := gocv.ContourArea(contour)

		if area > 20 && area < 2000 {
			points := gocv.NewPointVectorFromPoints(contour.ToPoints())
			rect := gocv.BoundingRect(points)
			points.Close()

			centerX := rect.Min.X + rect.Dx()/2
			centerY := rect.Min.Y + rect.Dy()/2

			// Convert to fuzzy descriptors
			sizeCategory := fba.getSpotSizeCategory(area, geometry)
			positionZone := fba.getPositionZone(float64(centerX), float64(centerY), geometry)

			marking := fmt.Sprintf("%s_dark_spot_%s", sizeCategory, positionZone)
			markings = append(markings, marking)
		}
	}

	return removeDuplicates(markings)
}

// detectColoredMarkings detects colored spots and markings with fuzzy color categories
func (fba *FishBiometricAnalyzer) detectColoredMarkings(hsv gocv.Mat, mask gocv.Mat, geometry FishBodyGeometry) []string {
	var markings []string

	// Detect red/orange markings
	redMarkings := fba.detectColorRange(hsv, mask, geometry, 0, 20, 100, 255, 100, 255, "red_accent")
	markings = append(markings, redMarkings...)

	// Detect yellow markings
	yellowMarkings := fba.detectColorRange(hsv, mask, geometry, 20, 40, 100, 255, 100, 255, "yellow_accent")
	markings = append(markings, yellowMarkings...)

	// Detect green markings
	greenMarkings := fba.detectColorRange(hsv, mask, geometry, 40, 80, 100, 255, 100, 255, "green_accent")
	markings = append(markings, greenMarkings...)

	// Detect blue markings
	blueMarkings := fba.detectColorRange(hsv, mask, geometry, 100, 130, 100, 255, 100, 255, "blue_accent")
	markings = append(markings, blueMarkings...)

	return removeDuplicates(markings)
}

// detectBrightSpots detects bright/light colored markings
func (fba *FishBiometricAnalyzer) detectBrightSpots(hsv gocv.Mat, mask gocv.Mat, geometry FishBodyGeometry) []string {
	var markings []string

	// Create threshold for bright regions (high value, low saturation for white/silver)
	lowerBound := gocv.NewMatFromScalar(gocv.NewScalar(0, 0, 200, 0), gocv.MatTypeCV8UC3)
	defer lowerBound.Close()
	upperBound := gocv.NewMatFromScalar(gocv.NewScalar(180, 100, 255, 0), gocv.MatTypeCV8UC3)
	defer upperBound.Close()

	brightMask := gocv.NewMat()
	defer brightMask.Close()
	gocv.InRange(hsv, lowerBound, upperBound, &brightMask)

	// Apply fish mask
	fishBrightMask := gocv.NewMat()
	defer fishBrightMask.Close()
	gocv.BitwiseAnd(brightMask, mask, &fishBrightMask)

	spotContours := gocv.FindContours(fishBrightMask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer spotContours.Close()

	for i := 0; i < spotContours.Size(); i++ {
		contour := spotContours.At(i)
		area := gocv.ContourArea(contour)

		if area > 30 && area < 1500 {
			points := gocv.NewPointVectorFromPoints(contour.ToPoints())
			rect := gocv.BoundingRect(points)
			points.Close()

			centerX := rect.Min.X + rect.Dx()/2
			centerY := rect.Min.Y + rect.Dy()/2

			sizeCategory := fba.getSpotSizeCategory(area, geometry)
			positionZone := fba.getPositionZone(float64(centerX), float64(centerY), geometry)

			marking := fmt.Sprintf("%s_bright_spot_%s", sizeCategory, positionZone)
			markings = append(markings, marking)
		}
	}

	return removeDuplicates(markings)
}

// detectColorRange detects markings in a specific HSV color range
func (fba *FishBiometricAnalyzer) detectColorRange(hsv gocv.Mat, mask gocv.Mat, geometry FishBodyGeometry,
	hMin, hMax, sMin, sMax, vMin, vMax int, colorName string) []string {

	var markings []string

	lowerBound := gocv.NewMatFromScalar(gocv.NewScalar(float64(hMin), float64(sMin), float64(vMin), 0), gocv.MatTypeCV8UC3)
	defer lowerBound.Close()
	upperBound := gocv.NewMatFromScalar(gocv.NewScalar(float64(hMax), float64(sMax), float64(vMax), 0), gocv.MatTypeCV8UC3)
	defer upperBound.Close()

	colorMask := gocv.NewMat()
	defer colorMask.Close()
	gocv.InRange(hsv, lowerBound, upperBound, &colorMask)

	fishColorMask := gocv.NewMat()
	defer fishColorMask.Close()
	gocv.BitwiseAnd(colorMask, mask, &fishColorMask)

	spotContours := gocv.FindContours(fishColorMask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer spotContours.Close()

	for i := 0; i < spotContours.Size(); i++ {
		contour := spotContours.At(i)
		area := gocv.ContourArea(contour)

		if area > 25 && area < 1000 {
			points := gocv.NewPointVectorFromPoints(contour.ToPoints())
			rect := gocv.BoundingRect(points)
			points.Close()

			centerX := rect.Min.X + rect.Dx()/2
			centerY := rect.Min.Y + rect.Dy()/2

			sizeCategory := fba.getSpotSizeCategory(area, geometry)
			positionZone := fba.getPositionZone(float64(centerX), float64(centerY), geometry)

			marking := fmt.Sprintf("%s_%s_%s", sizeCategory, colorName, positionZone)
			markings = append(markings, marking)
		}
	}

	return removeDuplicates(markings)
}

// getSpotSizeCategory categorizes spot size relative to fish body
func (fba *FishBiometricAnalyzer) getSpotSizeCategory(area float64, geometry FishBodyGeometry) string {
	bodyArea := geometry.majorAxis * geometry.minorAxis
	relativeArea := area / bodyArea

	if relativeArea < 0.001 {
		return "tiny"
	} else if relativeArea < 0.005 {
		return "small"
	} else if relativeArea < 0.02 {
		return "medium"
	} else if relativeArea < 0.05 {
		return "large"
	}
	return "major"
}

// getPositionZone categorizes position relative to fish body geometry
func (fba *FishBiometricAnalyzer) getPositionZone(x, y float64, geometry FishBodyGeometry) string {
	// Calculate relative position
	relativeX := (x - geometry.centerX) / geometry.majorAxis
	relativeY := (y - geometry.centerY) / geometry.minorAxis

	// Determine primary zone
	if math.Abs(relativeX) < 0.3 && math.Abs(relativeY) < 0.3 {
		return "center_body"
	}

	// Head/tail regions (along major axis)
	if relativeX < -0.3 {
		return "head_region"
	} else if relativeX > 0.3 {
		return "tail_region"
	}

	// Dorsal/ventral regions (along minor axis)
	if relativeY < -0.3 {
		return "dorsal_area"
	} else if relativeY > 0.3 {
		return "ventral_area"
	}

	return "mid_body"
}

// analyzeOverallColorPattern determines the fish's overall color family and transitions
func (fba *FishBiometricAnalyzer) analyzeOverallColorPattern(img gocv.Mat, mask gocv.Mat) []string {
	var patterns []string

	// Convert to HSV for color analysis
	hsv := gocv.NewMat()
	defer hsv.Close()
	gocv.CvtColor(img, &hsv, gocv.ColorBGRToHSV)

	// Apply mask to only analyze fish pixels
	maskedHSV := gocv.NewMat()
	defer maskedHSV.Close()
	hsv.CopyToWithMask(&maskedHSV, mask)

	// Split HSV channels
	channels := gocv.Split(maskedHSV)
	defer func() {
		for _, ch := range channels {
			ch.Close()
		}
	}()

	// Analyze hue distribution (color family)
	hueHist := gocv.NewMat()
	defer hueHist.Close()

	// Calculate hue histogram
	histSize := []int{36}          // 36 bins for 360 degrees / 10
	histRange := []float64{0, 180} // OpenCV HSV uses 0-180 for hue
	gocv.CalcHist([]gocv.Mat{channels[0]}, []int{0}, mask, &hueHist, histSize, histRange, false)

	// Find dominant hue ranges
	colorFamily := fba.getDominantColorFamily(hueHist)
	if colorFamily != "" {
		patterns = append(patterns, colorFamily)
	}

	// Analyze saturation to determine if colors are vivid or muted
	saturationPattern := fba.analyzeSaturationPattern(channels[1], mask)
	if saturationPattern != "" {
		patterns = append(patterns, saturationPattern)
	}

	// Analyze value (brightness) distribution for contrast patterns
	contrastPattern := fba.analyzeContrastPattern(channels[2], mask)
	if contrastPattern != "" {
		patterns = append(patterns, contrastPattern)
	}

	return removeDuplicates(patterns)
}

// detectSpottedPattern detects if the fish has a spotted pattern
func (fba *FishBiometricAnalyzer) detectSpottedPattern(img gocv.Mat, mask gocv.Mat) []string {
	var patterns []string

	// Convert to HSV
	hsv := gocv.NewMat()
	defer hsv.Close()
	gocv.CvtColor(img, &hsv, gocv.ColorBGRToHSV)

	// Look for circular/oval features (spots)
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)

	// Apply gaussian blur to reduce noise
	blurred := gocv.NewMat()
	defer blurred.Close()
	gocv.GaussianBlur(gray, &blurred, image.Pt(5, 5), 0, 0, gocv.BorderDefault)

	// Detect dark and light spots
	darkSpots := fba.countSpots(blurred, mask, true)   // dark spots
	lightSpots := fba.countSpots(blurred, mask, false) // light spots

	totalSpots := darkSpots + lightSpots
	if totalSpots > 15 {
		if totalSpots > 50 {
			patterns = append(patterns, "heavily_spotted")
		} else {
			patterns = append(patterns, "spotted")
		}

		// Add density info
		if darkSpots > lightSpots*2 {
			patterns = append(patterns, "dark_spotted_pattern")
		} else if lightSpots > darkSpots*2 {
			patterns = append(patterns, "light_spotted_pattern")
		}
	}

	return removeDuplicates(patterns)
}

// detectStripedPattern detects linear patterns (stripes)
func (fba *FishBiometricAnalyzer) detectStripedPattern(img gocv.Mat, mask gocv.Mat) []string {
	var patterns []string

	// Convert to grayscale
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)

	// Apply edge detection to find linear features
	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(gray, &edges, 50, 150)

	// Apply mask
	maskedEdges := gocv.NewMat()
	defer maskedEdges.Close()
	gocv.BitwiseAnd(edges, mask, &maskedEdges)

	// Use Hough Line Transform to detect linear patterns
	lines := gocv.NewMat()
	defer lines.Close()
	gocv.HoughLines(maskedEdges, &lines, 1, math.Pi/180, 50)

	lineCount := lines.Rows()
	if lineCount > 10 {
		// Analyze line orientations to determine stripe direction
		horizontalLines := 0
		verticalLines := 0

		for i := 0; i < lineCount; i++ {
			// Get line parameters (rho, theta)
			rho := lines.GetFloatAt(i, 0)
			theta := lines.GetFloatAt(i, 1)

			_ = rho // rho is distance from origin, not needed for orientation

			// Convert theta to degrees and categorize
			thetaDeg := theta * 180 / math.Pi
			if (thetaDeg >= 0 && thetaDeg <= 30) || (thetaDeg >= 150 && thetaDeg <= 180) {
				horizontalLines++
			} else if thetaDeg >= 60 && thetaDeg <= 120 {
				verticalLines++
			}
		}

		if horizontalLines > verticalLines*2 {
			patterns = append(patterns, "horizontal_striped")
		} else if verticalLines > horizontalLines*2 {
			patterns = append(patterns, "vertical_striped")
		} else if lineCount > 20 {
			patterns = append(patterns, "heavily_striped")
		} else {
			patterns = append(patterns, "striped")
		}
	}

	return removeDuplicates(patterns)
}

// detectTexturePattern analyzes overall texture characteristics
func (fba *FishBiometricAnalyzer) detectTexturePattern(img gocv.Mat, mask gocv.Mat) []string {
	var patterns []string

	// Convert to grayscale
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)

	// Apply mask
	maskedGray := gocv.NewMat()
	defer maskedGray.Close()
	gray.CopyToWithMask(&maskedGray, mask)

	// Analyze texture using edge density
	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(maskedGray, &edges, 50, 150)

	edgePixels := gocv.CountNonZero(edges)
	fishPixels := gocv.CountNonZero(mask)

	if fishPixels > 0 {
		edgeDensity := float64(edgePixels) / float64(fishPixels)

		if edgeDensity > 0.3 {
			patterns = append(patterns, "highly_textured")
		} else if edgeDensity > 0.1 {
			patterns = append(patterns, "moderately_textured")
		} else if edgeDensity < 0.05 {
			patterns = append(patterns, "smooth_texture")
		}
	}

	// Use standard deviation to measure texture variation
	meanMat := gocv.NewMat()
	defer meanMat.Close()
	stdDevMat := gocv.NewMat()
	defer stdDevMat.Close()
	gocv.MeanStdDev(maskedGray, &meanMat, &stdDevMat)

	// Extract standard deviation value from the matrix
	stdDevValue := stdDevMat.GetFloatAt(0, 0)
	if stdDevValue > 40 {
		patterns = append(patterns, "high_contrast_pattern")
	} else if stdDevValue < 15 {
		patterns = append(patterns, "uniform_coloring")
	}

	return removeDuplicates(patterns)
}

// Helper function to count spots in an image
func (fba *FishBiometricAnalyzer) countSpots(gray gocv.Mat, mask gocv.Mat, darkSpots bool) int {
	threshold := gocv.NewMat()
	defer threshold.Close()

	if darkSpots {
		// Threshold for dark spots
		gocv.Threshold(gray, &threshold, 80, 255, gocv.ThresholdBinaryInv)
	} else {
		// Threshold for light spots
		gocv.Threshold(gray, &threshold, 180, 255, gocv.ThresholdBinary)
	}

	// Apply fish mask
	maskedThreshold := gocv.NewMat()
	defer maskedThreshold.Close()
	gocv.BitwiseAnd(threshold, mask, &maskedThreshold)

	// Find contours
	contours := gocv.FindContours(maskedThreshold, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	spotCount := 0
	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)

		// Filter by size to identify actual spots
		if area > 30 && area < 500 {
			// Check circularity to confirm it's spot-like
			points := gocv.NewPointVectorFromPoints(contour.ToPoints())
			perimeter := gocv.ArcLength(points, true)
			points.Close()

			if perimeter > 0 {
				circularity := 4 * math.Pi * area / (perimeter * perimeter)
				if circularity > 0.3 { // Reasonably circular
					spotCount++
				}
			}
		}
	}

	return spotCount
}

// getDominantColorFamily analyzes hue histogram to determine color family
func (fba *FishBiometricAnalyzer) getDominantColorFamily(hueHist gocv.Mat) string {
	// Find the peak in the histogram
	minVal, maxVal, minLoc, maxLoc := gocv.MinMaxLoc(hueHist)
	_ = minVal
	_ = minLoc

	if maxVal < 100 { // Not enough color data
		return "neutral_tones"
	}

	peakHue := maxLoc.X * 10 // Convert back to hue degrees (0-180 -> 0-360)

	// Classify into color families
	if peakHue >= 0 && peakHue <= 30 || peakHue >= 330 {
		return "warm_red_tones"
	} else if peakHue > 30 && peakHue <= 90 {
		return "warm_yellow_tones"
	} else if peakHue > 90 && peakHue <= 150 {
		return "cool_green_tones"
	} else if peakHue > 150 && peakHue <= 270 {
		return "cool_blue_tones"
	} else {
		return "purple_magenta_tones"
	}
}

// analyzeSaturationPattern determines if colors are vivid or muted
func (fba *FishBiometricAnalyzer) analyzeSaturationPattern(satChannel gocv.Mat, mask gocv.Mat) string {
	meanMat := gocv.NewMat()
	defer meanMat.Close()
	stdDevMat := gocv.NewMat()
	defer stdDevMat.Close()
	gocv.MeanStdDev(satChannel, &meanMat, &stdDevMat)

	avgSaturation := meanMat.GetFloatAt(0, 0)

	if avgSaturation > 120 {
		return "vivid_coloring"
	} else if avgSaturation > 60 {
		return "moderate_coloring"
	} else {
		return "muted_coloring"
	}
}

// analyzeContrastPattern determines contrast characteristics
func (fba *FishBiometricAnalyzer) analyzeContrastPattern(valueChannel gocv.Mat, mask gocv.Mat) string {
	meanMat := gocv.NewMat()
	defer meanMat.Close()
	stdDevMat := gocv.NewMat()
	defer stdDevMat.Close()
	gocv.MeanStdDev(valueChannel, &meanMat, &stdDevMat)

	avgValue := meanMat.GetFloatAt(0, 0)
	valueVariation := stdDevMat.GetFloatAt(0, 0)

	if valueVariation > 50 {
		return "high_contrast"
	} else if valueVariation > 25 {
		return "moderate_contrast"
	} else if avgValue > 150 {
		return "bright_overall"
	} else if avgValue < 80 {
		return "dark_overall"
	} else {
		return "uniform_brightness"
	}
}

// getScarSizeCategory categorizes scar size relative to fish body
func (fba *FishBiometricAnalyzer) getScarSizeCategory(arcLength float64, geometry FishBodyGeometry) string {
	relativeLength := arcLength / geometry.majorAxis

	if relativeLength < 0.1 {
		return "minor"
	} else if relativeLength < 0.25 {
		return "medium"
	} else if relativeLength < 0.5 {
		return "major"
	}
	return "severe"
}

// getScarTypeFromShape determines scar type based on geometric properties
func (fba *FishBiometricAnalyzer) getScarTypeFromShape(rect image.Rectangle, arcLength, area float64) string {
	aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

	// Analyze shape characteristics
	if aspectRatio > 4.0 || aspectRatio < 0.25 {
		return "linear" // Very elongated, likely a cut or scratch
	} else if arcLength/area > 0.3 {
		return "jagged" // High perimeter to area ratio suggests irregular edges
	} else {
		return "rounded" // More compact, possibly bite mark or healed wound
	}
}

func removeDuplicates[T comparable](slice []T) []T {
	seen := make(map[T]bool)
	result := []T{}

	for _, item := range slice {
		if _, found := seen[item]; !found {
			seen[item] = true
			result = append(result, item)
		}
	}
	return result
}
