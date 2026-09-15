#!/usr/bin/env python
"""Create the Windows scheduled task that runs the pipeline every day.

    python Code/tools/setup_schedule.py                  # print, change nothing
    python Code/tools/setup_schedule.py --register       # create the task
    python Code/tools/setup_schedule.py --register --force   # replace one

Printing is the default. A run with no flags prints the command it would
issue and the task definition it would register, and touches nothing, so
the definition can be read before it exists anywhere.

Why an XML definition and not a one-line ``schtasks /Create /SC DAILY``:
the switches ``schtasks`` accepts cannot express *start the task as soon
as possible after a missed start*. A machine that is switched off at the
scheduled minute would simply skip that day, and the copy would quietly
fall behind by however many days it spent off. That behaviour is the
``StartWhenAvailable`` setting, which lives only in the task definition,
so the definition is generated here and registered with
``schtasks /Create /XML``.

What the task does: runs ``Code/update_data.py --daily`` with the
project folder as the working directory, once a day. The pipeline writes
its own log to ``Reports/logs/`` and ends it with an ALERT block naming
every condition it raised, so a task whose output nobody sees is still
readable afterwards - see the unattended-operation section of the
runbook.

This script runs on Windows. Anywhere else there is no ``schtasks`` to
call: it prints the command and the definition, says that nothing was
registered, and exits.
"""

import argparse
import datetime
import os
import platform
import subprocess
import sys
import tempfile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(THIS_DIR)
PROJECT_DIR = os.path.dirname(CODE_DIR)

DEFAULT_TASK_NAME = "retailAPOLLO daily refresh"
DEFAULT_TIME = "17:30"
# The pipeline's own runtime ceiling is minutes, not hours; a task still
# running after four is stuck rather than slow, and a stuck task holds
# the next day's start under MultipleInstancesPolicy IgnoreNew.
DEFAULT_TIME_LIMIT = "PT4H"

# Element order follows the file Task Scheduler itself exports, which is
# the order its schema expects. Reordering these is not cosmetic: the
# service rejects a definition whose elements arrive out of sequence.
TASK_XML = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{description}</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>{start_boundary}</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
{principal_user}      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>{wake}</WakeToRun>
    <ExecutionTimeLimit>{time_limit}</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{command}</Command>
      <Arguments>{arguments}</Arguments>
      <WorkingDirectory>{working_directory}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def _xml_escape(text):
    """``text`` as an XML text node.

    Folder names carry ampersands often enough to matter, and one of
    them turns the whole definition into a parse error the service
    reports as an unhelpful "the task XML is malformed".
    """
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _start_boundary(hhmm, today=None):
    """The trigger's first fire time, as Task Scheduler spells it.

    The date is tomorrow when the time of day has already passed, so
    registering the task at four in the afternoon does not count today's
    half-past-five as a start that was missed and run it on the spot.
    Local time with no zone suffix: the owner sets a wall-clock time and
    the task keeps it across daylight-saving changes.
    """
    hour, minute = [int(part) for part in hhmm.split(":")]
    today = today or datetime.datetime.now()
    first = today.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if first <= today:
        first += datetime.timedelta(days=1)
    return f"{first:%Y-%m-%dT%H:%M:%S}"


def _principal_user():
    """The ``<UserId>`` line for the task's principal, or ``""``.

    Named explicitly when the environment says who is logged on, so the
    definition reads as what it is. Left out otherwise, in which case
    the service registers the task for whoever runs the command.
    """
    user = os.environ.get("USERNAME", "").strip()
    domain = os.environ.get("USERDOMAIN", "").strip()
    if not user:
        return ""
    who = f"{domain}\\{user}" if domain else user
    return f"      <UserId>{_xml_escape(who)}</UserId>\n"


def build_xml(python_exe, script, workdir, hhmm=DEFAULT_TIME, wake=False,
              time_limit=DEFAULT_TIME_LIMIT, extra_args=("--daily",)):
    """The task definition, as the text that is handed to ``schtasks``.

    Args:
        python_exe: The interpreter that runs the pipeline. Every package
            the run needs has to be installed in this one.
        script: Path of ``Code/update_data.py``.
        workdir: Working directory for the run - the project folder.
        hhmm: Local time of day, ``HH:MM``.
        wake: Whether the machine is woken from sleep to run it.
        time_limit: How long a run may take before it is stopped.
        extra_args: Arguments after the script path.

    Returns:
        The definition as a string.
    """
    arguments = " ".join([f'"{script}"', *extra_args])
    return TASK_XML.format(
        description=_xml_escape(
            "Daily retail-sentiment refresh. Starts as soon as possible "
            "after a missed start, so a machine that was off still "
            "refreshes."),
        start_boundary=_start_boundary(hhmm),
        principal_user=_principal_user(),
        wake="true" if wake else "false",
        time_limit=_xml_escape(time_limit),
        command=_xml_escape(python_exe),
        arguments=_xml_escape(arguments),
        working_directory=_xml_escape(workdir))


def write_xml(path, xml):
    """Write the definition where ``schtasks`` can read it.

    UTF-16 with a byte-order mark, which is what the ``encoding="UTF-16"``
    declaration in the definition promises and what the service reads.
    Staged beside the target and swapped in, so an interrupted write
    cannot leave a half-written definition under a name the next command
    would hand to ``schtasks``.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-16") as fh:
        fh.write(xml)
    os.replace(tmp_path, path)


def _schtasks(args):
    """Run one ``schtasks`` command.

    Returns:
        ``(exit_code, output)``, or ``(None, reason)`` when the command
        does not exist on this machine - which is every machine that is
        not Windows, and is a thing to print rather than crash on.
    """
    try:
        r = subprocess.run(["schtasks", *args], capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def task_exists(name):
    """Whether a task of this name is registered.

    Returns ``(True/False, note)``, or ``(None, reason)`` when the
    question cannot be asked on this machine.
    """
    code, out = _schtasks(["/Query", "/TN", name])
    if code is None:
        return None, out
    return (code == 0), out.strip().splitlines()[-1] if out.strip() else ""


def main():
    """Print the task definition, and register it when asked."""
    ap = argparse.ArgumentParser(
        description="Create the daily scheduled task for the pipeline.")
    ap.add_argument("--task-name", default=DEFAULT_TASK_NAME,
                    help=f"name the task is registered under "
                         f"(default: {DEFAULT_TASK_NAME!r})")
    ap.add_argument("--time", default=DEFAULT_TIME,
                    help=f"local time of day, HH:MM (default: {DEFAULT_TIME})")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter the task runs the pipeline with "
                         "(default: the one running this script)")
    ap.add_argument("--args", default="--daily",
                    help="arguments passed to update_data.py "
                         "(default: --daily)")
    ap.add_argument("--time-limit", default=DEFAULT_TIME_LIMIT,
                    help=f"how long a run may take before the service "
                         f"stops it, as an ISO 8601 duration "
                         f"(default: {DEFAULT_TIME_LIMIT})")
    ap.add_argument("--wake", action="store_true",
                    help="wake the machine from sleep to run the task. Off "
                         "by default: a missed start is already recovered "
                         "as soon as the machine is available, and waking "
                         "depends on firmware and the power plan, so a "
                         "task that relies on it can look scheduled and "
                         "never fire")
    ap.add_argument("--xml-out", default=None,
                    help="where to write the task definition (default: a "
                         "file in the machine's temp folder)")
    ap.add_argument("--register", action="store_true",
                    help="actually create the task. Without it nothing on "
                         "this machine changes")
    ap.add_argument("--force", action="store_true",
                    help="replace a task of the same name. Without it an "
                         "existing task is left exactly as it is")
    ap.add_argument("--dry-run", action="store_true",
                    help="print and change nothing (the default)")
    args = ap.parse_args()

    script = os.path.join(CODE_DIR, "update_data.py")
    extra = tuple(a for a in args.args.split() if a)
    xml = build_xml(args.python, script, PROJECT_DIR, hhmm=args.time,
                    wake=args.wake, time_limit=args.time_limit,
                    extra_args=extra)
    xml_path = args.xml_out or os.path.join(tempfile.gettempdir(),
                                            "retailapollo_daily_task.xml")
    create = (f'schtasks /Create /TN "{args.task_name}" '
              f'/XML "{xml_path}"' + (" /F" if args.force else ""))

    print("=" * 72)
    print("DAILY SCHEDULED TASK")
    print(f"  task name  : {args.task_name}")
    print(f"  runs       : every day at {args.time} local time")
    print("  catch-up   : yes - StartWhenAvailable, so a start missed "
          "while the machine was off runs as soon as it is back")
    print(f"  wake       : {'yes' if args.wake else 'no'} (--wake turns it on)")
    print(f"  command    : {args.python}")
    print(f"  arguments  : {script} {' '.join(extra)}")
    print(f"  in folder  : {PROJECT_DIR}")
    print(f"  definition : {xml_path}")
    print("=" * 72)
    print()
    print("THE COMMAND")
    print(f"  {create}")
    print()
    print("THE DEFINITION")
    for line in xml.splitlines():
        print("  " + line)
    print()
    print("AFTERWARDS")
    print(f'  inspect : schtasks /Query /TN "{args.task_name}" /V /FO LIST')
    print(f'  run now : schtasks /Run /TN "{args.task_name}"')
    print(f'  pause   : schtasks /Change /TN "{args.task_name}" /DISABLE')
    print(f'  resume  : schtasks /Change /TN "{args.task_name}" /ENABLE')
    print(f'  delete  : schtasks /Delete /TN "{args.task_name}" /F')
    print()

    if not args.register:
        print("nothing was registered: printing is the default, and "
              "--register is what creates the task.")
        return 0

    if platform.system() != "Windows":
        print(f"NOT REGISTERED: schtasks is a Windows command and this is "
              f"{platform.system()}. The command and the definition above "
              f"are the whole of what has to run on the Windows machine.")
        return 1

    exists, note = task_exists(args.task_name)
    if exists is None:
        print(f"NOT REGISTERED: schtasks could not be run ({note}).")
        return 1
    if exists and not args.force:
        print(f'NOT REGISTERED: a task named "{args.task_name}" already '
              f"exists and is left exactly as it is. Read it with "
              f'schtasks /Query /TN "{args.task_name}" /V /FO LIST, then '
              f"pass --force to replace it, or --task-name to register "
              f"a second one.")
        return 1

    try:
        write_xml(xml_path, xml)
    except OSError as exc:
        print(f"NOT REGISTERED: the definition could not be written to "
              f"{xml_path} ({type(exc).__name__}: {exc}).")
        return 1

    code, out = _schtasks(["/Create", "/TN", args.task_name, "/XML", xml_path]
                          + (["/F"] if args.force else []))
    if code is None:
        print(f"NOT REGISTERED: schtasks could not be run ({out}).")
        return 1
    print(out.strip())
    if code != 0:
        print(f"NOT REGISTERED: schtasks exited {code}.")
        return 1
    print(f'REGISTERED: "{args.task_name}" runs every day at {args.time} '
          f"and catches up a missed start.")
    print(f'Check it with: schtasks /Query /TN "{args.task_name}" /V /FO LIST')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
