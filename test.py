import os
import win32gui
import time
# 查找NVIDIA控制面板程序路径
# 该路径可能因安装位置不同而不同
nv_path = ""
for root, dirs, files in os.walk("C:\\Program Files\\NVIDIA Corporation"):
    if "nvcplui.exe" in files:
        nv_path = os.path.join(root, "nvcplui.exe")
        break

# 调用NVIDIA控制面板程序并打开“系统信息”窗口
os.system(nv_path + " -open=help_info")

# 等待1秒，确保系统信息窗口完全打开
time.sleep(1)

# 查找“CUDA”项并输出版本号
hwnd = win32gui.FindWindow(None, "NVIDIA System Information")
if hwnd != 0:
    treeview_hwnd = win32gui.FindWindowEx(hwnd, None, "SysTreeView32", None)
    if treeview_hwnd != 0:
        cuda_item_hwnd = win32gui.FindWindowEx(treeview_hwnd, None, "SysTreeView32", "CUDA")
        if cuda_item_hwnd != 0:
            cuda_version = win32gui.GetDlgItemText(hwnd, 0x0294).split(":", 1)[1].strip()
            print("CUDA version:", cuda_version)
            win32gui.SendMessage(hwnd, win32con.WM_CLOSE, 0, 0)