import numpy as np

# ============================================================
# STUDENT PERFORMANCE ANALYTICS
# NumPy-based ML Data Preparation Project
# ============================================================

# ------------------------------------------------------------
# 1. DATASET
# ------------------------------------------------------------

students = np.array([
    "Srikar", "Rahul", "Priya", "Arjun", "Ananya",
    "Kiran", "Sneha", "Varun", "Meera", "Rohan"
])

subjects = np.array([
    "Math", "Science", "Python", "AI", "Statistics"
])

marks = np.array([
    [85, 92, 88, 90, 87],
    [72, 80, 78, 75, 82],
    [95, 94, 96, 92, 91],
    [61, 65, 68, 64, 70],
    [88, 91, 94, 90, 93],
    [76, 73, 80, 78, 75],
    [90, 87, 89, 91, 88],
    [68, 72, 70, 65, 74],
    [82, 85, 80, 84, 86],
    [55, 62, 58, 60, 65]
], dtype=np.float64)


# ------------------------------------------------------------
# 2. UNDERSTANDING THE DATA
# ------------------------------------------------------------

print("=" * 65)
print("       STUDENT PERFORMANCE ANALYTICS")
print("=" * 65)

print(f"\nNumber of students : {marks.shape[0]}")
print(f"Number of subjects : {marks.shape[1]}")
print(f"Array dimensions   : {marks.ndim}")
print(f"Total values       : {marks.size}")
print(f"Data type          : {marks.dtype}")
print(f"Memory used        : {marks.nbytes} bytes")


# ------------------------------------------------------------
# 3. DATA VALIDATION
# ------------------------------------------------------------

valid_marks = (marks >= 0) & (marks <= 100)

print("\nAll marks valid:", np.all(valid_marks))


# ------------------------------------------------------------
# 4. BASIC STATISTICS
# ------------------------------------------------------------

total_marks = np.sum(marks, axis=1)

average_marks = np.mean(marks, axis=1)

median_marks = np.median(marks, axis=1)

std_marks = np.std(marks, axis=1)

variance_marks = np.var(marks, axis=1)


# ------------------------------------------------------------
# 5. NORMALIZATION
# Convert marks from 0-100 to 0-1
# ------------------------------------------------------------

normalized_marks = marks / 100


# ------------------------------------------------------------
# 6. STANDARDIZATION
# ------------------------------------------------------------

standardized_marks = (
    marks - np.mean(marks, axis=0)
) / np.std(marks, axis=0)


# ------------------------------------------------------------
# 7. FEATURE ENGINEERING
# ------------------------------------------------------------

attendance = np.array([
    95, 88, 98, 72, 96,
    85, 94, 76, 91, 68
])

assignment_score = np.array([
    92, 81, 97, 65, 95,
    79, 90, 70, 88, 61
])


features = np.column_stack((
    average_marks,
    attendance,
    assignment_score
))


# ------------------------------------------------------------
# 8. WEIGHTED ML-STYLE PERFORMANCE SCORE
# ------------------------------------------------------------

weights = np.array([
    0.60,
    0.20,
    0.20
])

performance_score = features @ weights


# ------------------------------------------------------------
# 9. GRADE CLASSIFICATION
# ------------------------------------------------------------

grades = np.select(
    [
        performance_score >= 90,
        performance_score >= 80,
        performance_score >= 70,
        performance_score >= 60,
        performance_score >= 50
    ],
    [
        "A+",
        "A",
        "B",
        "C",
        "D"
    ],
    default="F"
)


# ------------------------------------------------------------
# 10. STUDENT REPORT
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("STUDENT PERFORMANCE REPORT")
print("-" * 65)

for i in range(len(students)):

    print(
        f"{students[i]:<10} | "
        f"Average: {average_marks[i]:>6.2f} | "
        f"Score: {performance_score[i]:>6.2f} | "
        f"Grade: {grades[i]}"
    )


# ------------------------------------------------------------
# 11. TOP AND LOWEST PERFORMER
# ------------------------------------------------------------

topper_index = np.argmax(performance_score)

lowest_index = np.argmin(performance_score)

print("\n" + "-" * 65)
print("PERFORMANCE INSIGHTS")
print("-" * 65)

print(
    f"Top Performer    : "
    f"{students[topper_index]}"
)

print(
    f"Lowest Performer : "
    f"{students[lowest_index]}"
)


# ------------------------------------------------------------
# 12. SUBJECT-WISE ANALYSIS
# ------------------------------------------------------------

subject_average = np.mean(marks, axis=0)

subject_highest = np.max(marks, axis=0)

subject_lowest = np.min(marks, axis=0)

print("\n" + "-" * 65)
print("SUBJECT-WISE ANALYSIS")
print("-" * 65)

for i in range(len(subjects)):

    print(
        f"{subjects[i]:<12} | "
        f"Average: {subject_average[i]:>6.2f} | "
        f"Highest: {subject_highest[i]:>3.0f} | "
        f"Lowest: {subject_lowest[i]:>3.0f}"
    )


# ------------------------------------------------------------
# 13. BOOLEAN FILTERING
# ------------------------------------------------------------

high_performers = average_marks >= 85

print("\nHigh-performing students:")

for student in students[high_performers]:

    print("-", student)


# ------------------------------------------------------------
# 14. WHERE / ANY / ALL
# ------------------------------------------------------------

improvement_needed = np.where(
    average_marks < 70
)[0]

high_achievers = np.any(
    marks > 90,
    axis=1
)

passed = np.all(
    marks >= 40,
    axis=1
)

print("\nStudents needing improvement:")

for index in improvement_needed:

    print("-", students[index])

print(
    "\nStudents who scored above 90 "
    "in at least one subject:",
    np.sum(high_achievers)
)

print(
    "Students who passed every subject:",
    np.sum(passed)
)


# ------------------------------------------------------------
# 15. RANKING
# ------------------------------------------------------------

ranking = np.argsort(
    performance_score
)[::-1]

print("\n" + "-" * 65)
print("STUDENT RANKING")
print("-" * 65)

for rank, index in enumerate(ranking, start=1):

    print(
        f"{rank:>2}. "
        f"{students[index]:<10} "
        f"{performance_score[index]:.2f}"
    )


# ------------------------------------------------------------
# 16. CLASS STATISTICS
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("CLASS STATISTICS")
print("-" * 65)

print(
    "Mean:",
    np.mean(marks)
)

print(
    "Median:",
    np.median(marks)
)

print(
    "Standard Deviation:",
    np.std(marks)
)

print(
    "90th Percentile:",
    np.percentile(marks, 90)
)


# ------------------------------------------------------------
# 17. MODERN NUMPY RANDOM GENERATION
# ------------------------------------------------------------

rng = np.random.default_rng(42)

sample_scores = rng.integers(
    50,
    101,
    size=5
)

print("\nSample generated scores:", sample_scores)


# ------------------------------------------------------------
# 18. SAVE NUMPY DATA
# ------------------------------------------------------------

np.save(
    "student_performance.npy",
    marks
)

print(
    "\nNumPy dataset saved successfully."
)


# ------------------------------------------------------------
# 19. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 65)

print(
    f"Students analyzed : {len(students)}"
)

print(
    f"Subjects analyzed : {len(subjects)}"
)

print(
    f"Class average     : {np.mean(marks):.2f}"
)

print(
    f"Top performer     : {students[topper_index]}"
)

print("=" * 65)