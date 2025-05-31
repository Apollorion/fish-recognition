package fish

import (
	"fish-recognition/pkg/cv"
	"math"
)

type Comparison struct {
	Fish1ID           string    `json:"fish1_id"`
	Fish2ID           string    `json:"fish2_id"`
	ConfidenceScore   float64   `json:"confidence_score"`
	SameFish          bool      `json:"same_fish"`
	MatchingFeatures  *Features `json:"matching_features"`
	DifferentFeatures *Features `json:"different_features"`
}

type Features struct {
	Markings      int     `json:"markings"`
	Scars         int     `json:"scars"`
	Patterns      int     `json:"patterns"`
	Features      int     `json:"features"`
	BodyRatioDiff float64 `json:"body_ratio_diff"`
	FinSimilarity float64 `json:"fin_similarity"`
	SameId        bool    `json:"same_id"`
}

func CompareFish(fish1, fish2 *cv.FishBiometric) *Comparison {
	markings, markingsDiff := compareStringSlices(fish1.UniqueMarkings, fish2.UniqueMarkings)
	scars, scarsDiff := compareStringSlices(fish1.ScarsAndDamage, fish2.ScarsAndDamage)
	patterns, patternsDiff := compareStringSlices(fish1.DistinctivePatterns, fish2.DistinctivePatterns)
	feat, diffFeat := compareStringSlices(fish1.PermanentFeatures, fish2.PermanentFeatures)

	matching := &Features{
		Markings:      markings,
		Scars:         scars,
		Patterns:      patterns,
		Features:      feat,
		BodyRatioDiff: math.Abs(fish1.NormalizedBodyRatio - fish2.NormalizedBodyRatio),
		FinSimilarity: compareFloatSlices(fish1.NormalizedFinShape, fish2.NormalizedFinShape),
		SameId:        fish1.ID == fish2.ID,
	}

	diff := &Features{
		Markings: markingsDiff,
		Scars:    scarsDiff,
		Patterns: patternsDiff,
		Features: diffFeat,
	}

	// Calculate overall confidence score
	confidence := calculateConfidenceScore(matching, diff, fish1, fish2)

	// Determine if it's the same fish (high confidence threshold)
	sameFish := confidence >= 75.0 || matching.SameId

	return &Comparison{
		Fish1ID:           fish1.ID,
		Fish2ID:           fish2.ID,
		ConfidenceScore:   confidence,
		SameFish:          sameFish,
		MatchingFeatures:  matching,
		DifferentFeatures: diff,
	}
}

func compareStringSlices(slice1, slice2 []string) (int, int) {
	matches := 0
	diffs := 0
	for _, item1 := range slice1 {
		found := false
		for _, item2 := range slice2 {
			if item1 == item2 {
				found = true
				matches++
				break
			}
		}
		if !found {
			diffs++
		}
	}
	return matches, diffs
}

func compareFloatSlices(slice1, slice2 []float64) float64 {
	if len(slice1) == 0 && len(slice2) == 0 {
		return 1.0
	}
	if len(slice1) == 0 || len(slice2) == 0 {
		return 0.0
	}

	minLen := len(slice1)
	if len(slice2) < minLen {
		minLen = len(slice2)
	}

	totalDiff := 0.0
	for i := 0; i < minLen; i++ {
		totalDiff += math.Abs(slice1[i] - slice2[i])
	}

	// Normalize by number of comparisons and convert to similarity
	avgDiff := totalDiff / float64(minLen)
	similarity := math.Max(0, 1.0-avgDiff)
	return similarity
}

func calculateConfidenceScore(mf *Features, df *Features, fish1, fish2 *cv.FishBiometric) float64 {

	confidence := 0.0

	// ID match gives very high confidence
	if mf.SameId {
		confidence += 60.0
	}

	// Calculate feature-specific weights based on match vs difference ratio
	
	// Unique markings are most valuable
	markingWeight := 0.015
	if mf.Markings > 0 {
		ratio := float64(mf.Markings) / math.Max(1.0, float64(df.Markings))
		if ratio > 2.0 {
			markingWeight *= 1.0 + (ratio-2.0)*0.3
		}
	}
	confidence += float64(mf.Markings) * markingWeight

	// Scars are extremely distinctive
	scarWeight := 0.020
	if mf.Scars > 0 {
		ratio := float64(mf.Scars) / math.Max(1.0, float64(df.Scars))
		if ratio > 2.0 {
			scarWeight *= 1.0 + (ratio-2.0)*0.3
		}
	}
	confidence += float64(mf.Scars) * scarWeight

	// Patterns are moderately valuable
	patternWeight := 0.008
	if mf.Patterns > 0 {
		ratio := float64(mf.Patterns) / math.Max(1.0, float64(df.Patterns))
		if ratio > 2.0 {
			patternWeight *= 1.0 + (ratio-2.0)*0.3
		}
	}
	confidence += float64(mf.Patterns) * patternWeight

	// Permanent features are somewhat valuable
	featureWeight := 0.005
	if mf.Features > 0 {
		ratio := float64(mf.Features) / math.Max(1.0, float64(df.Features))
		if ratio > 2.0 {
			featureWeight *= 1.0 + (ratio-2.0)*0.3
		}
	}
	confidence += float64(mf.Features) * featureWeight

	// Body ratio similarity
	if mf.BodyRatioDiff < 0.05 {
		confidence += 0.10
	} else if mf.BodyRatioDiff < 0.1 {
		confidence += 0.05
	}

	// Fin shape similarity
	confidence += mf.FinSimilarity * 0.10

	// Boost confidence if both fish have high individual confidence
	avgIndividualConfidence := (fish1.Confidence + fish2.Confidence) / 2.0
	if avgIndividualConfidence > 80.0 {
		confidence += 5.0
	}

	// Cap at 100%
	return math.Min(100.0, confidence)
}
