"""AXON Triage Module

Automated hypothesis selection and prioritization.
"""

import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from ..utils import load_config, ensure_dir

class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority enumeration."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """Task data structure."""
    id: str
    name: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    due_date: Optional[datetime] = None
    assigned_to: Optional[str] = None
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}

class TriageSystem:
    """Intelligent task triage and queue management system."""
    
    def __init__(self, config_path: str = "queue.yml"):
        self.config_path = Path(config_path)
        self.tasks: Dict[str, Task] = {}
        self.queue_history: List[Dict[str, Any]] = []
        self.config = load_config()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load existing queue if available
        self.load_queue()
    
    def load_queue(self) -> bool:
        """Load queue from YAML file."""
        try:
            if not self.config_path.exists():
                self.logger.info(f"Queue file {self.config_path} not found, starting with empty queue")
                return False
            
            with open(self.config_path, 'r') as f:
                queue_data = yaml.safe_load(f)
            
            if not queue_data:
                return False
            
            # Parse tasks from YAML
            tasks_data = queue_data.get('tasks', [])
            
            for task_data in tasks_data:
                try:
                    task = self._parse_task_from_dict(task_data)
                    self.tasks[task.id] = task
                except Exception as e:
                    self.logger.error(f"Failed to parse task {task_data.get('id', 'unknown')}: {e}")
                    continue
            
            self.logger.info(f"Loaded {len(self.tasks)} tasks from queue")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load queue: {e}")
            return False
    
    def save_queue(self) -> bool:
        """Save queue to YAML file."""
        try:
            ensure_dir(self.config_path.parent)
            
            # Convert tasks to serializable format
            tasks_data = []
            for task in self.tasks.values():
                task_dict = asdict(task)
                # Convert datetime objects to strings
                task_dict['created_at'] = task.created_at.isoformat()
                task_dict['updated_at'] = task.updated_at.isoformat()
                if task.due_date:
                    task_dict['due_date'] = task.due_date.isoformat()
                # Convert enums to strings
                task_dict['priority'] = task.priority.name
                task_dict['status'] = task.status.value
                
                tasks_data.append(task_dict)
            
            queue_data = {
                'version': '1.0',
                'updated_at': datetime.now().isoformat(),
                'tasks': tasks_data
            }
            
            with open(self.config_path, 'w') as f:
                yaml.dump(queue_data, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(f"Saved {len(self.tasks)} tasks to queue")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save queue: {e}")
            return False
    
    def _parse_task_from_dict(self, task_data: Dict[str, Any]) -> Task:
        """Parse task from dictionary."""
        # Parse datetime fields
        created_at = datetime.fromisoformat(task_data['created_at']) if isinstance(task_data['created_at'], str) else task_data['created_at']
        updated_at = datetime.fromisoformat(task_data['updated_at']) if isinstance(task_data['updated_at'], str) else task_data['updated_at']
        due_date = None
        if task_data.get('due_date'):
            due_date = datetime.fromisoformat(task_data['due_date']) if isinstance(task_data['due_date'], str) else task_data['due_date']
        
        # Parse enums
        priority = TaskPriority[task_data['priority']] if isinstance(task_data['priority'], str) else task_data['priority']
        status = TaskStatus(task_data['status']) if isinstance(task_data['status'], str) else task_data['status']
        
        return Task(
            id=task_data['id'],
            name=task_data['name'],
            description=task_data['description'],
            priority=priority,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            due_date=due_date,
            assigned_to=task_data.get('assigned_to'),
            dependencies=task_data.get('dependencies', []),
            metadata=task_data.get('metadata', {})
        )
    
    def add_task(self, 
                 name: str, 
                 description: str, 
                 priority: TaskPriority = TaskPriority.MEDIUM,
                 due_date: Optional[datetime] = None,
                 assigned_to: Optional[str] = None,
                 dependencies: List[str] = None,
                 metadata: Dict[str, Any] = None) -> str:
        """Add a new task to the queue."""
        
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.tasks)}"
        
        task = Task(
            id=task_id,
            name=name,
            description=description,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            due_date=due_date,
            assigned_to=assigned_to,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        self.logger.info(f"Added task: {name} (ID: {task_id})")
        
        return task_id
    
    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Update task status."""
        if task_id not in self.tasks:
            self.logger.error(f"Task {task_id} not found")
            return False
        
        old_status = self.tasks[task_id].status
        self.tasks[task_id].status = status
        self.tasks[task_id].updated_at = datetime.now()
        
        self.logger.info(f"Updated task {task_id} status: {old_status.value} -> {status.value}")
        return True
    
    def get_prioritized_queue(self) -> List[Task]:
        """Get tasks sorted by priority and other factors."""
        
        def priority_score(task: Task) -> Tuple[int, int, int]:
            """Calculate priority score for sorting."""
            
            # Primary: Priority level (higher is better)
            priority_val = task.priority.value
            
            # Secondary: Urgency based on due date
            urgency = 0
            if task.due_date:
                days_until_due = (task.due_date - datetime.now()).days
                if days_until_due < 0:
                    urgency = 100  # Overdue
                elif days_until_due <= 1:
                    urgency = 50   # Due soon
                elif days_until_due <= 7:
                    urgency = 25   # Due this week
            
            # Tertiary: Age of task (older tasks get slight priority)
            age_days = (datetime.now() - task.created_at).days
            age_score = min(age_days, 30)  # Cap at 30 days
            
            return (-priority_val, -urgency, -age_score)
        
        # Filter to pending tasks only
        pending_tasks = [task for task in self.tasks.values() if task.status == TaskStatus.PENDING]
        
        # Check dependencies
        ready_tasks = []
        for task in pending_tasks:
            if self._are_dependencies_met(task):
                ready_tasks.append(task)
        
        # Sort by priority score
        ready_tasks.sort(key=priority_score)
        
        return ready_tasks
    
    def _are_dependencies_met(self, task: Task) -> bool:
        """Check if all task dependencies are completed."""
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                self.logger.warning(f"Dependency {dep_id} not found for task {task.id}")
                return False
            
            if self.tasks[dep_id].status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next highest priority task."""
        queue = self.get_prioritized_queue()
        return queue[0] if queue else None
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        total_tasks = len(self.tasks)
        
        status_counts = {}
        priority_counts = {}
        
        for task in self.tasks.values():
            # Count by status
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by priority
            priority = task.priority.name
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # Calculate completion rate
        completed = status_counts.get('completed', 0)
        completion_rate = completed / total_tasks if total_tasks > 0 else 0
        
        # Find overdue tasks
        overdue_tasks = 0
        for task in self.tasks.values():
            if task.due_date and task.due_date < datetime.now() and task.status != TaskStatus.COMPLETED:
                overdue_tasks += 1
        
        return {
            'total_tasks': total_tasks,
            'status_breakdown': status_counts,
            'priority_breakdown': priority_counts,
            'completion_rate': completion_rate,
            'overdue_tasks': overdue_tasks,
            'ready_tasks': len(self.get_prioritized_queue())
        }
    
    def cleanup_completed_tasks(self, days_old: int = 30) -> int:
        """Remove completed tasks older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        tasks_to_remove = []
        for task_id, task in self.tasks.items():
            if (task.status == TaskStatus.COMPLETED and 
                task.updated_at < cutoff_date):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        
        self.logger.info(f"Cleaned up {len(tasks_to_remove)} old completed tasks")
        return len(tasks_to_remove)
    
    def export_report(self) -> Dict[str, Any]:
        """Export comprehensive triage report."""
        stats = self.get_task_statistics()
        queue = self.get_prioritized_queue()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'next_tasks': [
                {
                    'id': task.id,
                    'name': task.name,
                    'priority': task.priority.name,
                    'created_at': task.created_at.isoformat(),
                    'due_date': task.due_date.isoformat() if task.due_date else None
                }
                for task in queue[:10]  # Top 10 tasks
            ],
            'overdue_tasks': [
                {
                    'id': task.id,
                    'name': task.name,
                    'due_date': task.due_date.isoformat(),
                    'days_overdue': (datetime.now() - task.due_date).days
                }
                for task in self.tasks.values()
                if task.due_date and task.due_date < datetime.now() and task.status != TaskStatus.COMPLETED
            ]
        }
        
        return report

def create_sample_queue() -> TriageSystem:
    """Create a sample queue for demonstration."""
    triage = TriageSystem()
    
    # Add sample tasks
    triage.add_task(
        "Data Quality Check",
        "Validate synthetic data generation and check for anomalies",
        TaskPriority.HIGH,
        datetime.now() + timedelta(days=1)
    )
    
    triage.add_task(
        "Model Performance Review",
        "Analyze model performance metrics and identify improvement opportunities",
        TaskPriority.MEDIUM,
        datetime.now() + timedelta(days=3)
    )
    
    triage.add_task(
        "Feature Engineering Optimization",
        "Optimize feature engineering pipeline for better performance",
        TaskPriority.MEDIUM,
        datetime.now() + timedelta(days=7)
    )
    
    triage.add_task(
        "Risk Management Update",
        "Update risk management parameters based on recent market conditions",
        TaskPriority.CRITICAL,
        datetime.now() + timedelta(hours=12)
    )
    
    return triage

def main():
    """Main triage system demonstration."""
    print("=== AXON Triage System ===")
    
    # Load configuration
    config = load_config()
    ensure_dir("knowledge")
    
    try:
        # Initialize triage system
        triage = TriageSystem()
        
        # If no existing queue, create sample
        if not triage.tasks:
            print("No existing queue found, creating sample tasks...")
            triage = create_sample_queue()
        
        # Display current statistics
        stats = triage.get_task_statistics()
        print(f"\n[*] Queue Statistics:")
        print(f"  Total Tasks: {stats['total_tasks']}")
        print(f"  Ready Tasks: {stats['ready_tasks']}")
        print(f"  Completion Rate: {stats['completion_rate']:.1%}")
        print(f"  Overdue Tasks: {stats['overdue_tasks']}")

        # Show status breakdown
        print(f"\n[*] Status Breakdown:")
        for status, count in stats['status_breakdown'].items():
            print(f"  {status.title()}: {count}")

        # Show priority breakdown
        print(f"\n[*] Priority Breakdown:")
        for priority, count in stats['priority_breakdown'].items():
            print(f"  {priority}: {count}")

        # Show next tasks in queue
        queue = triage.get_prioritized_queue()
        if queue:
            print(f"\n[*] Next Tasks in Queue:")
            for i, task in enumerate(queue[:5], 1):
                due_str = f" (Due: {task.due_date.strftime('%Y-%m-%d')})" if task.due_date else ""
                print(f"  {i}. [{task.priority.name}] {task.name}{due_str}")

        # Get next task
        next_task = triage.get_next_task()
        if next_task:
            print(f"\n[*] Next Recommended Task:")
            print(f"  Name: {next_task.name}")
            print(f"  Priority: {next_task.priority.name}")
            print(f"  Description: {next_task.description}")
        
        # Save queue and export report
        triage.save_queue()
        
        report = triage.export_report()
        report_path = "knowledge/triage_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Emit a compact metrics artifact for smoke/automation
        ensure_dir("outputs/metrics")
        triage_metrics = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'total_tasks': report['statistics']['total_tasks'],
            'ready_tasks': report['statistics']['ready_tasks'],
            'overdue_tasks': report['statistics']['overdue_tasks'],
        }
        with open("outputs/metrics/TRIAGE.json", 'w') as f:
            json.dump(triage_metrics, f, indent=2)

        print(f"\n[SUCCESS] Triage system completed successfully!")
        print(f"Queue saved to: {triage.config_path}")
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        print(f"[ERROR] Triage system failed: {e}")
        raise

if __name__ == "__main__":
    main()
