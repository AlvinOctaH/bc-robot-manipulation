import rclpy
from rclpy.node import Node

import json
import time
import csv
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict

from std_msgs.msg import String
from geometry_msgs.msg import PoseArray


@dataclass
class TransferEvent:
    """Records timing and outcome of a single peg transfer."""
    peg_id:    int
    start_time: float
    end_time:   float = 0.0
    success:    bool  = False
    duration:   float = 0.0


@dataclass
class SessionMetrics:
    """Aggregated metrics for a full session (all 6 pegs)."""
    session_id:           str   = ''
    total_pegs:           int   = 0
    successful_transfers: int   = 0
    failed_transfers:     int   = 0
    total_time_s:         float = 0.0
    avg_transfer_time_s:  float = 0.0
    min_transfer_time_s:  float = 0.0
    max_transfer_time_s:  float = 0.0
    success_rate_pct:     float = 0.0
    events:               list  = field(default_factory=list)


class EvaluationNode(Node):
    """
    Records task performance and saves results to CSV + JSON.

    Why is this important for the portfolio?
      Prof. Hwang's peg transfer paper benchmarks against human
      surgeons — speed, accuracy, consistency. Having an evaluation
      node that produces real metrics shows you think like a
      researcher, not just a programmer.

    Output files saved to log_dir:
      session_<id>.json  — full event log for one run
      results.csv        — one row per session, easy to compare runs

    Topics:
      SUB  /task_state  (std_msgs/String)
      SUB  /arm_status  (std_msgs/String)
      SUB  /peg_poses   (geometry_msgs/PoseArray)
    """

    def __init__(self):
        super().__init__('evaluation')

        self.declare_parameter('log_dir',     '/tmp/peg_transfer_logs')
        self.declare_parameter('log_results', True)

        self.log_dir     = self.get_parameter('log_dir').value
        self.log_results = self.get_parameter('log_results').value

        os.makedirs(self.log_dir, exist_ok=True)

        # ── State ────────────────────────────────────────────────────────
        self.session = SessionMetrics(
            session_id=datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.events          = []
        self.current_event   = None
        self.session_start   = time.time()
        self.prev_task_state = ''

        # ── Subscribers ──────────────────────────────────────────────────
        self.task_sub = self.create_subscription(
            String,    '/task_state', self._task_state_callback, 10)
        self.arm_sub  = self.create_subscription(
            String,    '/arm_status', self._arm_status_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseArray, '/peg_poses',  self._peg_poses_callback,  10)

        # ── Timer: print live stats every 5s ────────────────────────────
        self.create_timer(5.0, self._print_live_stats)

        self.get_logger().info(
            f'Evaluation node started. Logs -> {self.log_dir}')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _task_state_callback(self, msg: String):
        state = msg.data

        # GRASPING transition → start timing this transfer
        if state == 'GRASPING' and self.prev_task_state != 'GRASPING':
            peg_id = len(self.events)
            self.current_event = TransferEvent(
                peg_id=peg_id, start_time=time.time())
            self.get_logger().debug(f'Timing peg {peg_id}')

        # RETREATING transition → transfer complete, stop timing
        if state == 'RETREATING' and self.prev_task_state != 'RETREATING':
            if self.current_event is not None:
                self.current_event.end_time = time.time()
                self.current_event.duration = (
                    self.current_event.end_time -
                    self.current_event.start_time)
                self.current_event.success = True
                self.events.append(self.current_event)
                self.get_logger().info(
                    f'Peg {self.current_event.peg_id} done in '
                    f'{self.current_event.duration:.2f}s')
                self.current_event = None

        # DONE → save full session
        if state == 'DONE' and self.prev_task_state != 'DONE':
            self._finalise_session()

        self.prev_task_state = state

    def _arm_status_callback(self, msg: String):
        """Record failed transfers."""
        if msg.data == 'FAILED' and self.current_event is not None:
            self.current_event.end_time = time.time()
            self.current_event.duration = (
                self.current_event.end_time -
                self.current_event.start_time)
            self.current_event.success = False
            self.events.append(self.current_event)
            self.get_logger().warn(
                f'Peg {self.current_event.peg_id} FAILED.')
            self.current_event = None

    def _peg_poses_callback(self, msg: PoseArray):
        self.session.total_pegs = len(msg.poses)

    # ── Metrics ───────────────────────────────────────────────────────────

    def _compute_metrics(self) -> SessionMetrics:
        m = self.session
        successes = [e for e in self.events if e.success]
        failures  = [e for e in self.events if not e.success]

        m.successful_transfers = len(successes)
        m.failed_transfers     = len(failures)
        m.total_time_s         = time.time() - self.session_start
        m.success_rate_pct     = (
            100.0 * len(successes) / len(self.events)
            if self.events else 0.0)

        if successes:
            durations              = [e.duration for e in successes]
            m.avg_transfer_time_s  = sum(durations) / len(durations)
            m.min_transfer_time_s  = min(durations)
            m.max_transfer_time_s  = max(durations)

        m.events = [asdict(e) for e in self.events]
        return m

    def _print_live_stats(self):
        m = self._compute_metrics()
        self.get_logger().info(
            f'[EVAL] {m.successful_transfers}/{m.total_pegs} transferred | '
            f'Success: {m.success_rate_pct:.0f}% | '
            f'Avg: {m.avg_transfer_time_s:.2f}s | '
            f'Total: {m.total_time_s:.0f}s')

    def _finalise_session(self):
        m = self._compute_metrics()
        self.get_logger().info(
            f'\n{"="*50}\n'
            f'SESSION COMPLETE\n'
            f'  Transferred : {m.successful_transfers}/{m.total_pegs}\n'
            f'  Success rate: {m.success_rate_pct:.1f}%\n'
            f'  Total time  : {m.total_time_s:.1f}s\n'
            f'  Avg/transfer: {m.avg_transfer_time_s:.2f}s\n'
            f'{"="*50}')

        if self.log_results:
            self._save_json(m)
            self._append_csv(m)

    def _save_json(self, m: SessionMetrics):
        path = os.path.join(self.log_dir, f'session_{m.session_id}.json')
        with open(path, 'w') as f:
            json.dump(asdict(m), f, indent=2)
        self.get_logger().info(f'Saved -> {path}')

    def _append_csv(self, m: SessionMetrics):
        path = os.path.join(self.log_dir, 'results.csv')
        write_header = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    'session_id', 'total_pegs', 'successful',
                    'failed', 'success_rate_pct', 'total_time_s',
                    'avg_time_s', 'min_time_s', 'max_time_s',
                ])
            writer.writerow([
                m.session_id, m.total_pegs,
                m.successful_transfers, m.failed_transfers,
                f'{m.success_rate_pct:.1f}', f'{m.total_time_s:.2f}',
                f'{m.avg_transfer_time_s:.2f}',
                f'{m.min_transfer_time_s:.2f}',
                f'{m.max_transfer_time_s:.2f}',
            ])
        self.get_logger().info(f'Appended -> {path}')


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()