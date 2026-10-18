# Giấy phép và nguồn ảnh BDD100K

Thư mục `data/real/pilot_images/` chứa **200 ảnh preview của đúng 100 ảnh pilot** M31, lấy từ bộ dữ liệu
**BDD100K** (tập 10k segmentation, split train). Không có ảnh, video hay nhãn BDD100K nào khác trong repository.

- `detection/<image_id>.jpg`: ảnh BDD100K kèm box `car` do YOLOv8n dự đoán (pre-label).
- `segmentation/<image_id>.jpg`: ảnh BDD100K kèm mask `car` do SAM-B dự đoán, prompt bằng box YOLOv8n.

**Chỉnh sửa so với dữ liệu gốc:** ảnh đã được vẽ đè box/mask, nhãn confidence và dòng tiêu đề (overlay do nhóm tạo
khi chạy pilot), rồi lưu sang JPEG chất lượng 92 (không lấy mẫu giảm màu), giữ nguyên độ phân giải 1280×720.
Tạo lại bằng `scripts/make_pilot_previews.py`.

**Mục đích sử dụng:** demo học tập / nghiên cứu, **phi lợi nhuận**, không thu phí, theo giấy phép dưới đây.
Giấy phép này **không** cấp quyền sử dụng thương mại cho người ngoài thành viên BDD / BAIR Commons; mọi sử dụng
thương mại cần liên hệ UC Berkeley Office of Technology Licensing như trong giấy phép.

Mã nguồn M31 trong repository này không thuộc giấy phép dưới đây; giấy phép chỉ áp dụng cho các ảnh BDD100K
(và ảnh chỉnh sửa từ chúng) trong thư mục này.

## Giấy phép dữ liệu BDD100K (nguyên văn)

Nguồn: trang giấy phép chính thức của BDD100K, tệp `doc/source/license.rst` trong
<https://github.com/bdd100k/bdd100k> (commit `6144873b6514`), áp dụng cho
"The data and labels downloaded from https://bdd-data.berkeley.edu/".

```text
Copyright ©2018. The Regents of the University of California (Regents). All Rights Reserved. 

THIS SOFTWARE AND/OR DATA WAS DEPOSITED IN THE BAIR OPEN RESEARCH COMMONS REPOSITORY ON 1/1/2021

Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement; and permission to use, copy, modify and distribute this software for commercial purposes (such rights not subject to transfer) to BDD and BAIR Commons members and their affiliates, is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
```

## Trích dẫn

Nếu sử dụng dữ liệu hoặc ảnh từ thư mục này, vui lòng trích dẫn bài báo BDD100K
(nguồn: README của <https://github.com/bdd100k/bdd100k>, commit `722cce8d95b6`):

```bibtex
@InProceedings{bdd100k,
    author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen,
              Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```

Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, Trevor Darrell.
*BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning.* CVPR 2020. arXiv:1805.04687.
