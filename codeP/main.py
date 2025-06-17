import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Unified CLI for Image Classification and Identification Project")

    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset folder inside DATA/")
    parser.add_argument('--model', type=str, required=True, help="Which task to run (e.g., classifier, segmenter, etc.)")
    parser.add_argument('--otherOptions', type=str, required=False, help="Optional subtask or setting (e.g., binary, live)")

    args = parser.parse_args()

    dataset_path = os.path.join("..", "DATA", args.dataset)
    model_name = args.model.lower()

    print("\nWelcome to our Image Classification and Identification Project")
    

    if model_name == 'classifier':
        import classifier; classifier.run(dataset_path, args.otherOptions)
    elif model_name == 'segmenter':
        import segmenter; segmenter.run(dataset_path, args.otherOptions)
    elif model_name == 'detector':
        import detector; detector.run(dataset_path, args.otherOptions)
    elif model_name == 'size_estimator':
        import size_estimator; size_estimator.run(dataset_path, args.otherOptions)
    elif model_name == 'digit_recognizer':
        import digit_recognizer; digit_recognizer.run(dataset_path, args.otherOptions)
    elif model_name == 'scanner':
        import scanner; scanner.run(dataset_path, args.otherOptions)
    elif model_name == 'omr':
        import omr_grader; omr_grader.run(dataset_path, args.otherOptions)
    elif model_name == 'ball_tracker':
        import ball_tracker; ball_tracker.run(dataset_path, args.otherOptions)
    elif model_name == 'drowsiness':
        import drowsiness; drowsiness.run(dataset_path, args.otherOptions)
    elif model_name == 'fracture':
        import fracture_detector; fracture_detector.run(dataset_path, args.otherOptions)
    else:
        print("❌ Unknown model/task. Please check your spelling.")

if __name__ == '__main__':
    main()
