class ApprovalManager:

    def __init__(self, approval_callback=None):
        self.approval_callback = approval_callback

    def request_approval(self, tool, arguments):
        if self.approval_callback is not None:
            return self.approval_callback(tool, arguments)

        print("\n" + "=" * 60)
        print("SECURITY APPROVAL REQUIRED")
        print("=" * 60)

        print(f"Tool        : {tool.name}")
        print(f"Category    : {tool.category}")
        print(f"Description : {tool.description}")
        print(f"Arguments   : {arguments}")

        print("=" * 60)

        answer = input(
            "Do you approve this operation? (yes/no): "
        ).strip().lower()

        if answer in ("yes", "y"):
            print("[APPROVAL] Granted.")
            return True

        print("[APPROVAL] Denied.")
        return False