def skill_gap(user_skills, required_skills):
    user = set(user_skills)
    required = set(required_skills.split(","))

    return {
        "matched_skills": list(user & required),
        "missing_skills": list(required - user)
    }
