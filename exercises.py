from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pathlib
import yaml


@dataclass
class Exercise:
    path: pathlib.Path
    name: str
    description: str
    problem_greeting: Optional[str] = None
    solution_greeting: Optional[str] = None
    problem_md: Optional[str] = None
    solution_md: str = ""
    test_code: Optional[str] = None
    code_samples: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, exercise_code: str) -> "Exercise":
        path = pathlib.Path("data/exercises") / exercise_code
        metadata_path = path / "metadata.yml"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing required file: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

        problem_md = None
        if (path / "problem.md").exists():
            problem_md = (path / "problem.md").read_text()

        solution_path = path / "solution.md"
        if solution_path.exists():
            solution_md = solution_path.read_text()
        else:
            solution_md = ""

        test_files = list(path.glob("test_*.py"))
        test_code = test_files[0].read_text() if test_files else None

        code_samples = {}
        if (path / "code").exists():
            for code_file in (path / "code").glob("*.py"):
                key = code_file.stem
                code_samples[key] = code_file.read_text()

        exercise = cls(
            path=path,
            name=metadata.pop("title"),
            description=metadata.pop("description"),
            problem_greeting=metadata.get("problem_greeting"),
            solution_greeting=metadata.get("solution_greeting"),
            problem_md=problem_md,
            solution_md=solution_md,
            test_code=test_code,
            code_samples=code_samples,
        )

        return exercise

    def __repr__(self) -> str:
        problem_preview = (self.problem_md[:25] + "...") if self.problem_md else None
        solution_preview = (self.solution_md[:25] + "...") if self.solution_md else None
        return f"Exercise(name='{self.name}', description='{self.description}', problem_md={problem_preview!r}, solution_md={solution_preview!r}, code_samples={len(self.code_samples)})"

    def get_code_sample(self, identifier: str) -> Optional[str]:
        return self.code_samples.get(identifier)

    def list_code_samples(self) -> List[str]:
        return sorted(self.code_samples.keys())
