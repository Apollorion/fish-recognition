package main

import (
	"encoding/json"
	"fmt"
	"os"

	"fish-recognition/pkg/cv"
	"fish-recognition/pkg/fish"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "fish-recognition",
	Short: "CLI tool for individual fish identification",
	Long:  "A CLI tool that uses biometric analysis to identify individual fish. Same fish = same ID, different fish = different ID.",
}

var identifyCmd = &cobra.Command{
	Use:   "identify [image-path]",
	Short: "Generate a unique identifier for an individual fish",
	Long:  "Analyzes a fish image to identify the specific individual based on scars, markings, and distinctive features. The same individual fish will always generate the same ID.",
	Args:  cobra.ExactArgs(1),
	RunE:  runIdentify,
}

var compareCmd = &cobra.Command{
	Use:   "compare [image-path-1] [image-path-2]",
	Short: "Compare two fish images and calculate similarity confidence",
	Long:  "Compares two fish images and provides a confidence score (0-100%) indicating how likely they are the same individual fish.",
	Args:  cobra.ExactArgs(2),
	RunE:  runCompare,
}

func init() {
	rootCmd.AddCommand(identifyCmd)
	rootCmd.AddCommand(compareCmd)
}

func runIdentify(cmd *cobra.Command, args []string) error {
	imagePath := args[0]

	// Create biometric fish analyzer for individual identification
	biometricAnalyzer := cv.NewFishBiometricAnalyzer()
	defer biometricAnalyzer.Close()

	// Identify individual fish using biometric analysis
	fishBiometric, err := biometricAnalyzer.IdentifyIndividualFish(imagePath)
	if err != nil {
		return fmt.Errorf("failed to identify individual fish: %w", err)
	}

	// Output results
	output, err := json.MarshalIndent(fishBiometric, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to format output: %w", err)
	}

	fmt.Println(string(output))
	return nil
}

func runCompare(cmd *cobra.Command, args []string) error {
	imagePath1 := args[0]
	imagePath2 := args[1]

	// Create biometric fish analyzer
	biometricAnalyzer := cv.NewFishBiometricAnalyzer()
	defer biometricAnalyzer.Close()

	// Analyze both fish
	fish1, err := biometricAnalyzer.IdentifyIndividualFish(imagePath1)
	if err != nil {
		return fmt.Errorf("failed to analyze first fish: %w", err)
	}

	fish2, err := biometricAnalyzer.IdentifyIndividualFish(imagePath2)
	if err != nil {
		return fmt.Errorf("failed to analyze second fish: %w", err)
	}

	// Compare the fish
	comparison := fish.CompareFish(fish1, fish2)

	// Output results
	output, err := json.MarshalIndent(comparison, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to format output: %w", err)
	}

	fmt.Println(string(output))
	return nil
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
